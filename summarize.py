"""Aggregate results/*.json into results/summary.md and results/f1_by_step.png (mean ± std over seeds)."""
import glob, json, os, re
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
runs = defaultdict(list)
for p in sorted(glob.glob(os.path.join(ROOT, 'results', '*.json'))):
    r = json.load(open(p))
    runs[re.sub(r'_s\d+', '', r['name'])].append(r)


def ms(vals, fmt='{:.3f}'):
    return f"{fmt.format(np.mean(vals))} ± {fmt.format(np.std(vals))}"


lines = ['## Main results (test time steps 35-49, illicit class, 3 seeds)', '',
         '| Model | Precision | Recall | F1 | PR-AUC | n seeds |', '|---|---|---|---|---|---|']
main = [('xgb', 'XGBoost (features only)'), ('mlp', 'MLP (features only)'),
        ('sage_f10-10', 'GraphSAGE, fan-out [10,10]'), ('sage_xgb_f10-10', 'XGBoost + GraphSAGE emb. (naive stacking)'),
        ('sage_xgb_oof_f10-10', 'XGBoost + GraphSAGE emb. (out-of-fold)')]
for key, label in main:
    rs = runs.get(key, [])
    if rs:
        t = [r['test'] for r in rs]
        lines.append(f"| {label} | {ms([x['precision'] for x in t])} | {ms([x['recall'] for x in t])} | "
                     f"{ms([x['f1'] for x in t])} | {ms([x['pr_auc'] for x in t])} | {len(rs)} |")

lines += ['', '## GraphSAGE fan-out trade-off (Mac M4 Pro CPU, 8 threads, batch 1024)', '',
          '| Fan-out | Test F1 | Test PR-AUC | s / epoch | sampling s / epoch | fwd+bwd s / epoch | epochs to early stop |',
          '|---|---|---|---|---|---|---|']
for key, label in [('sage_f5-5', '[5,5]'), ('sage_f10-10', '[10,10]'), ('sage_f25-10', '[25,10]'), ('sage_f-1--1', 'all neighbours')]:
    rs = runs.get(key, [])
    if rs:
        i = [r['info'] for r in rs]
        lines.append(f"| {label} | {ms([r['test']['f1'] for r in rs])} | {ms([r['test']['pr_auc'] for r in rs])} | "
                     f"{ms([x['t_epoch_mean'] for x in i], '{:.2f}')} | {ms([x['t_sample_mean'] for x in i], '{:.2f}')} | "
                     f"{ms([x['t_step_mean'] for x in i], '{:.2f}')} | {ms([x['epochs_run'] for x in i], '{:.0f}')} |")

emb = [r['info']['embedding'] for r in runs.get('sage_xgb_f10-10', [])]
if emb:
    lines += ['', f"Batch embedding generation (all {203769:,} nodes, full neighbours, batch 4096): "
                  f"{ms([e['seconds'] for e in emb], '{:.2f}')} s, {ms([e['nodes_per_sec'] / 1e3 for e in emb], '{:.0f}')}k nodes/s"]
open(os.path.join(ROOT, 'results', 'summary.md'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))

plt.figure(figsize=(7, 3.5))
for key, label in main:
    rs = runs.get(key, [])
    if rs:
        steps = sorted(int(k) for k in rs[0]['test_f1_by_step'])
        f1 = np.mean([[r['test_f1_by_step'][str(s)] for s in steps] for r in rs], 0)
        plt.plot(steps, f1, marker='o', ms=3, label=label)
plt.axvline(43, color='gray', ls='--', lw=1)
plt.text(43.2, 0.95, 'dark market\nshutdown', fontsize=8, color='gray', va='top')
plt.xlabel('test time step'); plt.ylabel('illicit F1 (mean of 3 seeds)'); plt.ylim(0, 1)
plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'results', 'f1_by_step.png'), dpi=150)
