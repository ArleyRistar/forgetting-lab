#!/usr/bin/env bash
# One process per (arm, seed) — memory grows across model loads inside a single
# process and OOM'd phase 2 at ~20 runs. Marker files, never pgrep.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$(git rev-parse --show-toplevel)"
log() { echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p outputs/phase2b
rm -f outputs/phase2b/CHAIN-{DONE,FAILED}
: > outputs/phase2b-chain.log
for seed in 0 1 2; do
  for arm in ternary float; do
    log "START $arm/s$seed"
    if uv run scripts/phase2b.py --arms "$arm" --seeds "$seed" \
         >> outputs/phase2b-chain.log 2>&1; then
      log "OK    $arm/s$seed  ($(df -h /home | tail -1 | awk '{print $4}') free)"
    else
      log "FAIL  $arm/s$seed"; touch outputs/phase2b/CHAIN-FAILED; exit 1
    fi
  done
done
log complete
touch outputs/phase2b/CHAIN-DONE
