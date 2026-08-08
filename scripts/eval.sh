#!/usr/bin/env bash
# Evaluate SmolLM2-360M with lm-evaluation-harness.
# Usage: scripts/eval.sh <run-tag> [peft-adapter-path]
set -euo pipefail
TAG=$1
PEFT=${2:+,peft=$2}
uv run lm_eval --model hf \
  --model_args "pretrained=HuggingFaceTB/SmolLM2-360M,dtype=bfloat16${PEFT}" \
  --tasks arc_easy,hellaswag,ifeval \
  --batch_size auto \
  --log_samples \
  --output_path "outputs/eval/${TAG}"
