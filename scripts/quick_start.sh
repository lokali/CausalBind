#!/usr/bin/env bash
# Quick start: download the released CausalBind checkpoints and reproduce the
# DUD-E and LIT-PCBA results of Table 1 on a single GPU.
#
# Usage: bash scripts/quick_start.sh [variant ...]     (default: sp lr emb atom)
# Requires the test benchmarks in test_datasets/ (see DATA.md) and the
# pretrained backbones in pretrain/ (bash scripts/download_data.sh pretrain).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_DIR}"

variants=("$@")
[ ${#variants[@]} -eq 0 ] && variants=(sp lr emb atom)
CKPT_URL="${CKPT_URL:-https://github.com/lokali/CausalBind/releases/download/v1.0}"
RESULTS="${RESULTS:-results/quick_start}"
mkdir -p checkpoints "${RESULTS}"

for v in "${variants[@]}"; do
  ckpt="checkpoints/causalbind_${v}.pt"
  if [ ! -s "${ckpt}" ]; then
    echo ">>> downloading ${ckpt}"
    wget -nv -O "${ckpt}" "${CKPT_URL}/causalbind_${v}.pt"
  fi
  mkdir -p "${RESULTS}/${v}"
  for task in DUDE PCBA; do
    echo ">>> evaluating CausalBind-${v} on ${task}"
    bash scripts/test.sh "${v}" "${task}" "${ckpt}" "${RESULTS}/${v}" > "${RESULTS}/${v}/${task}.log" 2>&1 \
      || { echo "evaluation failed, see ${RESULTS}/${v}/${task}.log"; exit 1; }
  done
done

python tools/summarize_results.py "${RESULTS}"
