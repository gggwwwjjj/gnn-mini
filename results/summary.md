## Main results (test time steps 35-49, illicit class, 3 seeds)

| Model | Precision | Recall | F1 | PR-AUC | n seeds |
|---|---|---|---|---|---|
| XGBoost (features only) | 0.796 ± 0.015 | 0.734 ± 0.001 | 0.764 ± 0.007 | 0.792 ± 0.001 | 3 |
| MLP (features only) | 0.501 ± 0.016 | 0.512 ± 0.027 | 0.506 ± 0.020 | 0.350 ± 0.043 | 3 |
| GraphSAGE, fan-out [10,10] | 0.569 ± 0.020 | 0.572 ± 0.015 | 0.570 ± 0.017 | 0.547 ± 0.033 | 3 |
| XGBoost + GraphSAGE emb. (naive stacking) | 0.576 ± 0.022 | 0.586 ± 0.020 | 0.581 ± 0.020 | 0.608 ± 0.039 | 3 |
| XGBoost + GraphSAGE emb. (out-of-fold) | 0.796 ± 0.111 | 0.701 ± 0.027 | 0.740 ± 0.039 | 0.776 ± 0.002 | 3 |

## GraphSAGE fan-out trade-off (Mac M4 Pro CPU, 8 threads, batch 1024)

| Fan-out | Test F1 | Test PR-AUC | s / epoch | sampling s / epoch | fwd+bwd s / epoch | epochs to early stop |
|---|---|---|---|---|---|---|
| [5,5] | 0.565 ± 0.010 | 0.542 ± 0.030 | 0.30 ± 0.00 | 0.05 ± 0.00 | 0.26 ± 0.00 | 51 ± 16 |
| [10,10] | 0.570 ± 0.017 | 0.547 ± 0.033 | 0.37 ± 0.00 | 0.05 ± 0.00 | 0.32 ± 0.00 | 35 ± 2 |
| [25,10] | 0.548 ± 0.008 | 0.512 ± 0.016 | 0.41 ± 0.03 | 0.06 ± 0.00 | 0.35 ± 0.03 | 40 ± 9 |
| all neighbours | 0.554 ± 0.019 | 0.509 ± 0.033 | 0.61 ± 0.00 | 0.07 ± 0.00 | 0.54 ± 0.00 | 38 ± 9 |

Batch embedding generation (all 203,769 nodes, full neighbours, batch 4096): 0.30 ± 0.07 s, 715 ± 147k nodes/s
