#!/usr/bin/env bash
# Shared model configurations for the CausalBind variants reported in the paper.
# Usage: source scripts/variant_config.sh <sp|lr|emb|atom|hypseek>
#   sets ${arch} and ${model_flags}

case "$1" in
  sp)   # CausalBind-SP: sparse concept-pair mask (Table 1, K=128, d_c=1024)
    arch="causal_three_hybrid_v2"
    model_flags="--num-concepts 128 --concept-dim 1024 --concept-layers 4 --sparsity-weight 1e-4 --sparsity-threshold 0.5"
    ;;
  lr)   # CausalBind-LR: low-rank concept-axis mask (Table 1, K=64, r=1)
    arch="causal_three_hybrid_v2"
    model_flags="--num-concepts 64 --concept-dim 1024 --concept-layers 4 --sparsity-weight 1e-4 --sparsity-threshold 0.5 --mask-type low_rank --mask-rank 1"
    ;;
  emb)  # CausalBind-EMB: constrained mask in pooled embedding space (Table 1, K=64, d_c=256)
    arch="causal_three_hybrid_v1"
    model_flags="--num-concepts 64 --concept-dim 256 --concept-layers 4 --sparsity-weight 1e-2 --sparsity-threshold 0.5"
    ;;
  atom|atomattn)  # CausalBind-ATOM: atom-attentive CausalBind-SP (Table 1; interpretability case study, App. A6.4)
    arch="causal_three_hybrid_v2_atomattn"
    model_flags="--num-concepts 128 --concept-dim 1024 --concept-layers 4 --sparsity-weight 1e-4 --sparsity-threshold 0.5"
    ;;
  hypseek)  # reproduced HypSeek baseline with the same frozen ESM-2 setup
    arch="three_hybrid_model_frozen"
    model_flags=""
    ;;
  *)
    echo "Unknown variant '$1' (expected: sp | lr | emb | atom | hypseek)" >&2
    return 1 2>/dev/null || exit 1
    ;;
esac
