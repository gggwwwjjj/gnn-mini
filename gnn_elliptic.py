"""Illicit-transaction detection on the Elliptic Bitcoin graph.

Models (same features, same temporal split):
  xgb      XGBoost on node features only
  mlp      MLP on node features only
  sage     GraphSAGE (PyG), mini-batch training with NeighborLoader
  sage_xgb XGBoost on node features + GraphSAGE embeddings (naive stacking: train-node embeddings
           come from a GNN that was fit on those same labels)
  sage_xgb_oof  same, but train-node embeddings are out-of-fold: training time steps are split into
           K folds and each fold is embedded by a GNN trained on the other folds

Split (Weber et al. 2019 test split): train time steps 1-29, val 30-34, test 35-49.
Every run writes its config, split sizes and metrics to results/<name>.json.
"""
import argparse, faulthandler, json, os, subprocess, sys, tempfile, time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score
from torch_geometric.datasets import EllipticBitcoinDataset
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected

faulthandler.enable()
ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_END, VAL_END = 29, 34


def load_data():
    ds = EllipticBitcoinDataset(os.path.join(ROOT, 'data/elliptic'))
    data = ds[0]
    # PyG drops the time step; read it back from the raw features file (column 1, same row order).
    raw = pd.read_csv(ds.raw_paths[0], header=None, usecols=[1])
    data.time = torch.from_numpy(raw[1].values)
    # Point-in-time check: no edge connects two different time steps, so a node's
    # neighbourhood never contains future transactions.
    src, dst = data.edge_index
    assert (data.time[src] == data.time[dst]).all()
    data.edge_index = to_undirected(data.edge_index)  # see both upstream and downstream flows

    labeled = data.y != 2
    data.train_idx = torch.where(labeled & (data.time <= TRAIN_END))[0]
    data.val_idx = torch.where(labeled & (data.time > TRAIN_END) & (data.time <= VAL_END))[0]
    data.test_idx = torch.where(labeled & (data.time > VAL_END))[0]

    # Standardise with statistics from the training period only.
    tr = data.time <= TRAIN_END
    mu, sd = data.x[tr].mean(0), data.x[tr].std(0).clamp_min(1e-6)
    data.x = (data.x - mu) / sd
    return data


def metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    return dict(precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred),
                f1=f1_score(y, pred), pr_auc=average_precision_score(y, p), threshold=float(thr))


def best_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[np.argmax(f1[:-1])])


def evaluate(data, p_val, p_test):
    """Pick the threshold on val, report test metrics overall and per time step."""
    yv, yt = data.y[data.val_idx].numpy(), data.y[data.test_idx].numpy()
    thr = best_threshold(yv, p_val)
    out = dict(val=metrics(yv, p_val, thr), test=metrics(yt, p_test, thr))
    ts = data.time[data.test_idx].numpy()
    out['test_f1_by_step'] = {int(t): f1_score(yt[ts == t], (p_test[ts == t] >= thr).astype(int), zero_division=0)
                              for t in np.unique(ts)}
    return out


# ---------------- XGBoost ----------------
def run_xgb(data, X, seed):
    """Runs xgb_worker.py in a subprocess (see the note there)."""
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = os.path.join(tmp, 'in.npz'), os.path.join(tmp, 'out.npz')
        np.savez(inp, Xtr=X[data.train_idx], ytr=data.y[data.train_idx].numpy(), Xv=X[data.val_idx],
                 yv=data.y[data.val_idx].numpy(), Xte=X[data.test_idx], seed=seed)
        subprocess.run([sys.executable, os.path.join(ROOT, 'xgb_worker.py'), inp, out], check=True)
        r = np.load(out)
        return r['pv'], r['pt'], dict(best_iteration=int(r['best_iteration']))


# ---------------- MLP / GraphSAGE ----------------
class MLP(torch.nn.Module):
    def __init__(self, d_in, hid, dropout):
        super().__init__()
        self.l1, self.l2, self.head, self.dropout = torch.nn.Linear(d_in, hid), torch.nn.Linear(hid, hid), torch.nn.Linear(hid, 2), dropout

    def forward(self, x, edge_index=None):
        x = F.dropout(F.relu(self.l1(x)), self.dropout, self.training)
        return self.head(F.dropout(F.relu(self.l2(x)), self.dropout, self.training))


class SAGE(torch.nn.Module):
    def __init__(self, d_in, hid, dropout):
        super().__init__()
        self.c1, self.c2, self.head, self.dropout = SAGEConv(d_in, hid), SAGEConv(hid, hid), torch.nn.Linear(hid, 2), dropout

    def embed(self, x, edge_index):
        x = F.dropout(F.relu(self.c1(x, edge_index)), self.dropout, self.training)
        return F.relu(self.c2(x, edge_index))

    def forward(self, x, edge_index):
        return self.head(F.dropout(self.embed(x, edge_index), self.dropout, self.training))


def make_loader(data, idx, fanout, batch_size, shuffle, num_workers=0):
    return NeighborLoader(data, num_neighbors=fanout, input_nodes=idx, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers)


@torch.no_grad()
def predict(model, data, idx, graph, args):
    """Full-neighbour inference in mini-batches (same path as batch embedding generation)."""
    model.eval()
    if not graph:
        return torch.softmax(model(data.x[idx]), 1)[:, 1].numpy()
    out = []
    for b in make_loader(data, idx, [-1, -1], 4096, False):
        out.append(torch.softmax(model(b.x, b.edge_index)[:b.batch_size], 1)[:, 1])
    return torch.cat(out).numpy()


def run_nn(data, args, seed, graph, train_idx=None):
    train_idx = data.train_idx if train_idx is None else train_idx
    torch.manual_seed(seed); np.random.seed(seed)
    model = (SAGE if graph else MLP)(data.num_features, args.hidden, args.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    w = torch.tensor([args.licit_weight, 1 - args.licit_weight])
    yv = data.y[data.val_idx].numpy()
    if graph:
        loader = make_loader(data, train_idx, args.fanout, args.batch_size, True, args.num_workers)
    best, best_state, bad, ep_log = -1, None, 0, []
    for ep in range(args.epochs):
        model.train()
        t_sample = t_step = 0.0
        t0 = time.perf_counter()
        if graph:
            it = iter(loader)
            while True:
                ts = time.perf_counter()
                try:
                    b = next(it)  # neighbour sampling + feature gather
                except StopIteration:
                    break
                t1 = time.perf_counter(); t_sample += t1 - ts
                opt.zero_grad()
                loss = F.cross_entropy(model(b.x, b.edge_index)[:b.batch_size], b.y[:b.batch_size], weight=w)
                loss.backward(); opt.step()
                t_step += time.perf_counter() - t1
        else:
            perm = train_idx[torch.randperm(len(train_idx))]
            for i in range(0, len(perm), args.batch_size):
                idx = perm[i:i + args.batch_size]
                t1 = time.perf_counter()
                opt.zero_grad()
                loss = F.cross_entropy(model(data.x[idx]), data.y[idx], weight=w)
                loss.backward(); opt.step()
                t_step += time.perf_counter() - t1
        t_epoch = time.perf_counter() - t0
        ap = average_precision_score(yv, predict(model, data, data.val_idx, graph, args))
        ep_log.append(dict(epoch=ep, val_pr_auc=ap, t_epoch=t_epoch, t_sample=t_sample, t_step=t_step))
        if ap > best:
            best, bad, best_state = ap, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    pv = predict(model, data, data.val_idx, graph, args)
    pt = predict(model, data, data.test_idx, graph, args)
    n = len(ep_log)
    timing = dict(epochs_run=n, best_val_pr_auc=best,
                  t_epoch_mean=sum(e['t_epoch'] for e in ep_log) / n,
                  t_sample_mean=sum(e['t_sample'] for e in ep_log) / n,
                  t_step_mean=sum(e['t_step'] for e in ep_log) / n)
    return model, pv, pt, timing


@torch.no_grad()
def generate_embeddings(model, data, out_path, meta):
    """Batch embedding generation for every node; writes a versioned file next to its config."""
    model.eval()
    t0 = time.perf_counter()
    embs = []
    for b in make_loader(data, torch.arange(data.num_nodes), [-1, -1], 4096, False):
        embs.append(model.embed(b.x, b.edge_index)[:b.batch_size])
    emb = torch.cat(embs)
    dt = time.perf_counter() - t0
    torch.save(dict(embedding=emb, meta=meta), out_path)
    return emb, dict(seconds=dt, nodes_per_sec=data.num_nodes / dt, dim=emb.shape[1], path=os.path.relpath(out_path, ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['xgb', 'mlp', 'sage', 'sage_xgb', 'sage_xgb_oof'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fanout', type=int, nargs='+', default=[10, 10])
    ap.add_argument('--batch_size', type=int, default=1024)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=5e-4)
    ap.add_argument('--licit_weight', type=float, default=0.3)  # 0.3/0.7 class weights as in Weber et al.
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--name', default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    name = args.name or f"{args.model}_s{args.seed}" + (f"_f{'-'.join(map(str, args.fanout))}" if 'sage' in args.model else '')

    t0 = time.perf_counter()
    data = load_data()
    print(f'[{name}] data loaded in {time.perf_counter() - t0:.1f}s: train {len(data.train_idx)} val {len(data.val_idx)} test {len(data.test_idx)}', flush=True)
    res = dict(name=name, config=vars(args), torch=torch.__version__,
               split=dict(train_steps=f'1-{TRAIN_END}', val_steps=f'{TRAIN_END + 1}-{VAL_END}', test_steps=f'{VAL_END + 1}-49',
                          n_train=len(data.train_idx), n_val=len(data.val_idx), n_test=len(data.test_idx)))
    t0 = time.perf_counter()
    X = data.x.numpy()
    if args.model == 'xgb':
        pv, pt, info = run_xgb(data, X, args.seed)
    elif args.model in ('mlp', 'sage'):
        _, pv, pt, info = run_nn(data, args, args.seed, graph=args.model == 'sage')
    else:  # sage_xgb, sage_xgb_oof
        model, _, _, info = run_nn(data, args, args.seed, graph=True)
        os.makedirs(os.path.join(ROOT, 'embeddings'), exist_ok=True)
        emb, info['embedding'] = generate_embeddings(model, data, os.path.join(ROOT, 'embeddings', f'{name}.pt'),
                                                     meta=dict(config=vars(args), train_steps=f'1-{TRAIN_END}'))
        if args.model == 'sage_xgb_oof':
            # Edges never cross time steps, so folds by time step share no edges.
            fold = data.time % args.folds
            for k in range(args.folds):
                held = data.train_idx[fold[data.train_idx] == k]
                m_k, _, _, _ = run_nn(data, args, args.seed * 100 + k, graph=True,
                                      train_idx=data.train_idx[fold[data.train_idx] != k])
                m_k.eval()
                with torch.no_grad():
                    for b in make_loader(data, held, [-1, -1], 4096, False):
                        emb[b.n_id[:b.batch_size]] = m_k.embed(b.x, b.edge_index)[:b.batch_size]
        pv, pt, xinfo = run_xgb(data, np.hstack([X, emb.numpy()]), args.seed)
        info['xgb'] = xinfo
    res['info'] = info
    res.update(evaluate(data, pv, pt))
    res['wall_seconds'] = time.perf_counter() - t0
    os.makedirs(os.path.join(ROOT, 'results'), exist_ok=True)
    with open(os.path.join(ROOT, 'results', f'{name}.json'), 'w') as f:
        json.dump(res, f, indent=1)
    t = res['test']
    print(f"[{name}] test illicit P {t['precision']:.3f} R {t['recall']:.3f} F1 {t['f1']:.3f} PR-AUC {t['pr_auc']:.3f} | {res['wall_seconds']:.0f}s", flush=True)


if __name__ == '__main__':
    main()
