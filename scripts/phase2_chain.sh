#!/usr/bin/env bash
# Phase-2 chain: seed 0 -> predictors -> prune -> seeds 1,2 -> predictors -> prune.
#
# It DRIVES the runs rather than waiting for someone else's, and never uses
# pgrep. The first attempt waited on `pgrep -f "scripts/phase2.py"`, which matched
# a lingering shell of the agent's own tooling whose command line contained that
# string, so it waited ~4h for a process that had already died. Exit codes and a
# marker file cannot cross-match anything.
#
# expandable_segments: the first attempt OOM'd on its 8th float run with 16 MiB
# free, and the allocator itself suggests this.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/arley/forgetting-lab
log() { echo "[$(date +%H:%M:%S)] $*"; }

step() {  # step <label> <cmd...>
  local label=$1; shift
  log "START $label"
  if "$@" >> outputs/phase2-chain.log 2>&1; then
    log "OK    $label"
  else
    log "FAIL  $label (exit $?) — chain stops here"
    touch outputs/phase2/CHAIN-FAILED
    exit 1
  fi
}

mkdir -p outputs/phase2
step "seed 0"            uv run scripts/phase2.py --seeds 0
step "predictors s0"     uv run scripts/phase2_flips.py --prune
log "disk: $(df -h /home | tail -1 | awk '{print $4" free, "$5" used"}')"
step "seeds 1,2"         uv run scripts/phase2.py --seeds 1,2
step "predictors s1,s2"  uv run scripts/phase2_flips.py --prune
log "complete: $(python3 -c "import json;print(len(json.load(open('outputs/phase2/results.json'))),'results,',len(json.load(open('outputs/phase2/predictors.json'))),'predictors')" 2>/dev/null)"
touch outputs/phase2/CHAIN-COMPLETE
