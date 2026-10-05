"""Quantify the sparse-antichain condition on learned CausalBind masks.

For each K x K mask in a checkpoint, this script applies the same effective
mask and hard gate used by the submitted V2 model:

    M_eff = tanh(M_raw) + 1
    support = (M_eff > threshold)

It then evaluates the row and column support families from Condition 2.i.
A pair violates the antichain condition when either support is contained in
the other; equal supports therefore also count as violations.

Reported diagnostics:
  * comparable-pair rate: fraction of unordered support pairs that violate;
  * equal/strict-containment decomposition of that rate;
  * exact maximum-antichain width (Dilworth's theorem), after collapsing
    duplicate supports because at most one duplicate can be retained;
  * support-size and hard-zero summaries.

Usage:
    python tools/analyze_antichain.py CHECKPOINT [--threshold 0.5]
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Iterable

import torch


def _hopcroft_karp(adjacency: list[list[int]], n_right: int) -> int:
    """Maximum bipartite matching size for left adjacency lists."""
    n_left = len(adjacency)
    pair_left = [-1] * n_left
    pair_right = [-1] * n_right
    distance = [0] * n_left
    infinity = n_left + n_right + 1

    def bfs() -> bool:
        queue: deque[int] = deque()
        found = False
        for left in range(n_left):
            if pair_left[left] == -1:
                distance[left] = 0
                queue.append(left)
            else:
                distance[left] = infinity
        while queue:
            left = queue.popleft()
            for right in adjacency[left]:
                next_left = pair_right[right]
                if next_left == -1:
                    found = True
                elif distance[next_left] == infinity:
                    distance[next_left] = distance[left] + 1
                    queue.append(next_left)
        return found

    def dfs(left: int) -> bool:
        for right in adjacency[left]:
            next_left = pair_right[right]
            if next_left == -1 or (
                distance[next_left] == distance[left] + 1 and dfs(next_left)
            ):
                pair_left[left] = right
                pair_right[right] = left
                return True
        distance[left] = infinity
        return False

    matching = 0
    while bfs():
        for left in range(n_left):
            if pair_left[left] == -1 and dfs(left):
                matching += 1
    return matching


def _to_bitsets(support: torch.Tensor) -> list[int]:
    bitsets = []
    for row in support.to(torch.bool).cpu().tolist():
        value = 0
        for index, active in enumerate(row):
            if active:
                value |= 1 << index
        bitsets.append(value)
    return bitsets


def support_family_stats(support: torch.Tensor) -> dict[str, float | int | bool]:
    """Compute exact containment and width statistics for one support family."""
    sets = _to_bitsets(support)
    count = len(sets)
    total_pairs = count * (count - 1) // 2
    comparable = 0
    equal = 0
    strict = 0
    for first in range(count):
        a = sets[first]
        for second in range(first + 1, count):
            b = sets[second]
            if a == b:
                comparable += 1
                equal += 1
            elif (a & ~b) == 0 or (b & ~a) == 0:
                comparable += 1
                strict += 1

    unique_sets = sorted(set(sets))
    # Strict-subset comparability graph on unique supports.  By Dilworth,
    # width = number of poset elements - maximum bipartite matching.
    adjacency: list[list[int]] = [[] for _ in unique_sets]
    for left, a in enumerate(unique_sets):
        for right, b in enumerate(unique_sets):
            if left != right and a != b and (a & ~b) == 0:
                adjacency[left].append(right)
    matching = _hopcroft_karp(adjacency, len(unique_sets))
    width = len(unique_sets) - matching

    sizes = sorted(value.bit_count() for value in sets)
    median = (
        float(sizes[count // 2])
        if count % 2
        else (sizes[count // 2 - 1] + sizes[count // 2]) / 2.0
    )
    return {
        "num_supports": count,
        "num_unique_supports": len(unique_sets),
        "unique_support_rate": len(unique_sets) / count,
        "support_size_min": min(sizes),
        "support_size_median": median,
        "support_size_max": max(sizes),
        "empty_supports": sum(size == 0 for size in sizes),
        "full_supports": sum(size == support.shape[1] for size in sizes),
        "comparable_pairs": comparable,
        "comparable_pair_rate": comparable / total_pairs if total_pairs else 0.0,
        "equal_support_pairs": equal,
        "equal_support_pair_rate": equal / total_pairs if total_pairs else 0.0,
        "strict_containment_pairs": strict,
        "strict_containment_pair_rate": strict / total_pairs if total_pairs else 0.0,
        "maximum_antichain_width": width,
        "maximum_antichain_fraction": width / count,
        "exact_antichain": comparable == 0,
    }


def _mask_parameters(state: dict[str, torch.Tensor]) -> Iterable[tuple[str, torch.Tensor]]:
    suffix = "_sparse_mask.mask_logits"
    for key, value in sorted(state.items()):
        if key.endswith(suffix) and value.ndim == 2:
            branch = key[: -len(suffix)]
            yield branch, value.float()


def analyze(checkpoint: Path, threshold: float) -> dict:
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = loaded["model"]
    output = {
        "checkpoint": str(checkpoint),
        "threshold": threshold,
        "branches": {},
    }
    stream_supports: dict[str, torch.Tensor] = {}
    for branch, raw_mask in _mask_parameters(state):
        effective = torch.tanh(raw_mask) + 1.0
        support = effective > threshold
        stream_supports[branch] = support
        output["branches"][f"stream/{branch}"] = {
            "shape": list(support.shape),
            "hard_zero_count": int((~support).sum()),
            "hard_zero_rate": float((~support).float().mean()),
            "rows": support_family_stats(support),
            "columns": support_family_stats(support.T),
        }
    # Match the paper's Table 3 convention.  The implementation applies a
    # separate mask to each tower, so the effective cross-modal support is the
    # element-wise intersection (equivalently, the support of their product).
    for protein_branch, pair_name in (
        ("pocket", "mol-pocket"),
        ("protein", "mol-sequence"),
    ):
        if "mol" not in stream_supports or protein_branch not in stream_supports:
            continue
        support = stream_supports["mol"] & stream_supports[protein_branch]
        output["branches"][f"pair/{pair_name}"] = {
            "shape": list(support.shape),
            "hard_zero_count": int((~support).sum()),
            "hard_zero_rate": float((~support).float().mean()),
            "rows": support_family_stats(support),
            "columns": support_family_stats(support.T),
        }
    if not output["branches"]:
        raise RuntimeError("No 2-D V2 sparse-mask parameters found")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print one tab-separated summary row per branch instead of JSON",
    )
    args = parser.parse_args()

    result = analyze(args.checkpoint, args.threshold)
    rendered = json.dumps(result, indent=2)
    if args.compact:
        print(
            "checkpoint\tbranch\tzero_rate\trow_violation_rate\tcolumn_violation_rate"
            "\trow_width\tcolumn_width\trow_unique\tcolumn_unique"
        )
        for branch, values in result["branches"].items():
            rows = values["rows"]
            columns = values["columns"]
            print(
                f"{args.checkpoint.parent.parent.name}\t{branch}"
                f"\t{values['hard_zero_rate']:.6f}"
                f"\t{rows['comparable_pair_rate']:.6f}"
                f"\t{columns['comparable_pair_rate']:.6f}"
                f"\t{rows['maximum_antichain_width']}"
                f"\t{columns['maximum_antichain_width']}"
                f"\t{rows['num_unique_supports']}"
                f"\t{columns['num_unique_supports']}"
            )
    else:
        print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
