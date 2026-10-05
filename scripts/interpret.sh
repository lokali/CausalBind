#!/usr/bin/env bash
# Local-to-global interpretability case study with the atom-attentive checkpoint
# (paper App. A6.4). Produces report.json and local_binding.{png,pdf} per target.
# Usage: bash scripts/interpret.sh <atomattn_checkpoint.pt> <output_dir> [targets]
set -euo pipefail

ckpt="$1"
out_dir="$2"
targets="${3:-thrb}"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "${REPO_DIR}/scripts/variant_config.sh" atom
DATA_ROOT="${TEST_DATA_ROOT:-${REPO_DIR}/test_datasets}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

python "${REPO_DIR}/tools/interpret_local_binding.py" "${DATA_ROOT}" \
    --user-dir "${REPO_DIR}/unimol" \
    --task test_task \
    --loss three_hybrid_loss \
    --arch ${arch} \
    ${model_flags} \
    --max-pocket-atoms 511 \
    --batch-size 1 \
    --num-workers 0 \
    --seed 1 \
    --fp16 \
    --device-id 0 \
    --path "${ckpt}" \
    --case-targets "${targets}" \
    --case-output "${out_dir}"
