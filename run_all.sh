#!/bin/bash
# Reproduces every result in README: 4 models x 3 seeds, then the fan-out sweep for GraphSAGE.
set -e
cd "$(dirname "$0")"
PY="uv run python -W ignore gnn_elliptic.py"
for s in 0 1 2; do
  for m in xgb mlp sage sage_xgb sage_xgb_oof; do $PY --model $m --seed $s; done
  for f in "5 5" "25 10" "-1 -1"; do $PY --model sage --seed $s --fanout $f; done
done
uv run python summarize.py
