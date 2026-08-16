#!/usr/bin/env bash
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(git rev-parse --show-toplevel)"
for arm in ternary float; do
  echo "[$(date +%H:%M:%S)] retrain $arm conflict s0 budget 300"
  uv run scripts/phase2.py --arms "$arm" --pairs conflict --seeds 0 --budgets 300 \
    >> outputs/h2-retrain.log 2>&1 || { echo "FAIL $arm"; touch outputs/H2-RETRAIN-FAILED; exit 1; }
done
touch outputs/H2-RETRAIN-DONE
echo "[$(date +%H:%M:%S)] retrain complete"
