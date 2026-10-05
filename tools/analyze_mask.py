"""
Post-hoc analysis of V2 SparseMask statistics.

Loads a V2 checkpoint, extracts the 3 branches' K×K masks
(mol_sparse_mask, pocket_sparse_mask, protein_sparse_mask), applies the
tanh-based sigmoid to get σ(M), and reports:

  - mean / std / min / max of σ(M)
  - sparsity rate at τ ∈ {0.01, 0.05, 0.1, 0.5}
  - effective rank at 90%, 95%, 99% variance
  - L1 norm and Frobenius norm

Usage:
    python analyze_mask.py <save_dir>
"""
import sys, os, json
import torch
import numpy as np


def sigma(x, mask_type="tanh_plus_1"):
    """Apply the same σ(·) as the model's forward."""
    if mask_type == "tanh_plus_1":
        return torch.tanh(x) + 1.0
    else:
        raise NotImplementedError(mask_type)


def effective_rank(M, variance_thresholds=(0.90, 0.95, 0.99)):
    """Compute the effective rank by cumulative singular-value variance."""
    _, s, _ = torch.linalg.svd(M.float())
    s2 = s ** 2
    cum = torch.cumsum(s2, dim=0) / torch.sum(s2)
    results = {}
    for thr in variance_thresholds:
        r = int((cum < thr).sum()) + 1
        results[f"r@{int(thr*100)}%"] = r
    return results, s.tolist()


def analyze_mask(ckpt_path, mask_type="tanh_plus_1"):
    """Load ckpt, find sparse-mask parameters, report stats.

    Handles two parameterizations:
      - V2 dense:    `<branch>.mask_logits` of shape (K, K)
      - V3 low-rank: `<branch>.mask_U` and `<branch>.mask_V` of shape (K, r)
                     → realized as M = U @ V.T
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"]

    # Discover branches and detect which parameterization is in use per branch.
    branches = sorted(set(
        k.split(".")[0].replace("_sparse_mask", "")
        for k in state.keys() if "sparse_mask" in k
    ))

    out = {}
    for branch in branches:
        u_key = f"{branch}_sparse_mask.mask_U"
        v_key = f"{branch}_sparse_mask.mask_V"
        l_key = f"{branch}_sparse_mask.mask_logits"

        # Prefer the low-rank pair if present and 2D (skip the alias m_logits in V3).
        if u_key in state and v_key in state and state[u_key].ndim == 2:
            U = state[u_key].float()
            V = state[v_key].float()
            M_raw = U @ V.t()  # (K, K), rank ≤ r
            r_constr = U.shape[1]
            mode = f"low_rank (r={r_constr})"
        elif l_key in state and state[l_key].ndim == 2:
            M_raw = state[l_key].float()
            mode = "dense"
        else:
            continue

        M_sig = sigma(M_raw, mask_type)    # effective mask σ(M)
        K = M_sig.shape[0]

        # Scalar stats
        stats = {
            "branch": branch,
            "mode": mode,
            "shape": list(M_sig.shape),
            "mean(sigma_M)": float(M_sig.mean()),
            "std(sigma_M)": float(M_sig.std()),
            "min(sigma_M)": float(M_sig.min()),
            "max(sigma_M)": float(M_sig.max()),
            "median(sigma_M)": float(M_sig.median()),
            "L1": float(M_sig.abs().sum()),
            "Frob": float(M_sig.norm()),
        }
        # Sparsity at thresholds
        for tau in (0.1, 0.5):
            stats[f"sparsity<{tau}"] = float((M_sig < tau).float().mean())

        # Effective rank (only for square K×K matrices, V2 case)
        if M_sig.ndim == 2 and M_sig.shape[0] == M_sig.shape[1]:
            er, _ = effective_rank(M_sig)
            stats.update(er)

        out[branch] = stats

    return out


def format_row(stats):
    return (
        f"{stats['branch']:<8} "
        f"mean={stats['mean(sigma_M)']:.4f} "
        f"std={stats['std(sigma_M)']:.4f} "
        f"min={stats['min(sigma_M)']:.4f} "
        f"max={stats['max(sigma_M)']:.4f} "
        f"median={stats['median(sigma_M)']:.4f} | "
        f"sp<0.1={stats.get('sparsity<0.1', 0):.3f} "
        f"sp<0.5={stats.get('sparsity<0.5', 0):.3f} | "
        f"r@90={stats.get('r@90%', '-')} "
        f"r@95={stats.get('r@95%', '-')} "
        f"r@99={stats.get('r@99%', '-')}"
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_mask.py <save_dir_or_ckpt_path> [--ckpt last|best_bedroc|best_auc]")
        sys.exit(1)
    path = sys.argv[1]

    which = "last"
    if "--ckpt" in sys.argv:
        which = sys.argv[sys.argv.index("--ckpt") + 1]

    if os.path.isfile(path):
        ckpt_path = path
    elif os.path.isdir(path):
        sd = path if path.endswith("savedir") else os.path.join(path, "savedir")
        if which == "last":
            ckpt_path = os.path.join(sd, "checkpoint_last.pt")
        elif which == "best_bedroc":
            ckpt_path = os.path.join(sd, "checkpoint_best.pt")
        elif which == "best_auc":
            # find epoch with max valid_auc from training log
            tlog = os.path.join(os.path.dirname(sd.rstrip("/savedir")), "..", "train_log")
            print(f"best_auc mode: not implemented standalone. Using checkpoint_last.pt")
            ckpt_path = os.path.join(sd, "checkpoint_last.pt")
    else:
        print(f"Not found: {path}")
        sys.exit(1)

    print(f"Loading {ckpt_path}")
    stats = analyze_mask(ckpt_path)
    if not stats:
        print("No sparse_mask parameters found — maybe not a V2 model?")
        sys.exit(1)
    print("-" * 130)
    for branch, s in stats.items():
        print(format_row(s))
    print("-" * 130)

    # Also emit JSON for aggregation
    out_json = ckpt_path.replace(".pt", ".mask_stats.json")
    with open(out_json, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
