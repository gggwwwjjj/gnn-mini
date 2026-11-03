"""XGBoost in its own process: torch and xgboost ship different libomp builds that segfault in one process on macOS."""
import faulthandler, sys

import numpy as np
import xgboost as xgb

faulthandler.enable()
d = np.load(sys.argv[1])
ytr = d['ytr']
clf = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=6, subsample=0.8,
                        colsample_bytree=0.8, tree_method='hist', eval_metric='aucpr',
                        early_stopping_rounds=100, scale_pos_weight=(ytr == 0).sum() / (ytr == 1).sum(),
                        random_state=int(d['seed']), n_jobs=8)
clf.fit(d['Xtr'], ytr, eval_set=[(d['Xv'], d['yv'])], verbose=False)
np.savez(sys.argv[2], pv=clf.predict_proba(d['Xv'])[:, 1], pt=clf.predict_proba(d['Xte'])[:, 1],
         best_iteration=clf.best_iteration)
