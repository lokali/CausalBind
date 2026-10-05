#!/usr/bin/env bash
# Evaluate a checkpoint on DUD-E, LIT-PCBA, or FEP.
# Usage: bash scripts/test.sh <sp|lr|emb|atom|hypseek> <DUDE|PCBA|FEP> <checkpoint.pt> <results_dir>
# The paper reports the last checkpoint (checkpoint_last.pt) of each run.
set -euo pipefail

variant="$1"
TASK="$2"
weight_path="$3"
results_path="$4"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "${REPO_DIR}/scripts/variant_config.sh" "${variant}"

DATA_ROOT="${TEST_DATA_ROOT:-${REPO_DIR}/test_datasets}"
export PYTHONPATH="${REPO_DIR}/unimol:${PYTHONPATH:-}"
mkdir -p "${results_path}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python "${REPO_DIR}/unimol/test.py" "${DATA_ROOT}" \
    --user-dir "${REPO_DIR}/unimol" \
    --valid-subset test \
    --results-path "${results_path}" \
    --num-workers 0 --ddp-backend c10d \
    --distributed-world-size 1 \
    --batch-size 128 \
    --task test_task \
    --loss three_hybrid_loss \
    --arch ${arch} \
    ${model_flags} \
    --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
    --seed 1 \
    --path "${weight_path}" \
    --log-interval 100 \
    --log-format simple \
    --max-pocket-atoms 511 \
    --test-task ${TASK}
