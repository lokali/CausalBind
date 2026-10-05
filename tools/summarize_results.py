#!/usr/bin/env python3
"""Summarize the DUD-E and LIT-PCBA results written by scripts/quick_start.sh.

Expects logs named <results_root>/<variant>/<DUDE|PCBA>.log.

Usage: python tools/summarize_results.py results/quick_start
"""

import re
import sys
from pathlib import Path

VARIANTS = {"sp": "CausalBind-SP", "lr": "CausalBind-LR", "emb": "CausalBind-EMB", "atom": "CausalBind-ATOM"}
PATTERNS = (r"^auc mean ([\d.]+)", r"^bedroc mean ([\d.]+)", r"^ef 0\.01 mean ([\d.]+)")


def read_metrics(log):
    if not log.exists():
        return None
    text = log.read_text(errors="ignore")
    values = [re.search(p, text, re.M) for p in PATTERNS]
    return tuple(float(m.group(1)) for m in values) if all(values) else None


def fmt(values):
    return " / ".join(f"{v:.2f}" if i == 2 else f"{v:.3f}" for i, v in enumerate(values)) if values else "not found"


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/quick_start")
    print(f"{'Method':17s} | {'DUD-E AUROC / BEDROC / EF@1%':30s} | {'LIT-PCBA AUROC / BEDROC / EF@1%':30s}")
    print("-" * 84)
    for variant, name in VARIANTS.items():
        if not (root / variant).exists():
            continue
        dude = read_metrics(root / variant / "DUDE.log")
        pcba = read_metrics(root / variant / "PCBA.log")
        print(f"{name:17s} | {fmt(dude):30s} | {fmt(pcba):30s}")


if __name__ == "__main__":
    main()
