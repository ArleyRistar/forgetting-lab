#!/usr/bin/env bash
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(git rev-parse --show-toplevel)"
for arm in ternary float; do
  echo "[$(date +%H:%M:%S)] B-only $arm"
  uv run scripts/h2_bonly.py --arm $arm >> outputs/h2-bonly.log 2>&1 \
    || { touch outputs/H2-BONLY-FAILED; exit 1; }
done
touch outputs/H2-BONLY-DONE
