#!/usr/bin/env python3
"""Export a training checkpoint as a compact inference checkpoint.

Keeps only the model weights (dropping optimizer state and the training
arguments) and stores them in float16. Evaluation runs the model in fp16
(`--fp16` casts the model with `model.half()`), so the exported checkpoint
gives the same scores as the original one.

Usage:
    python tools/export_checkpoint.py save/causalbind_sp_seed1/savedir/checkpoint_last.pt \
        checkpoints/causalbind_sp.pt --variant sp
"""

import argparse
import hashlib

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("--variant", required=True, help="variant name in scripts/variant_config.sh")
    args = parser.parse_args()

    state = torch.load(args.src, map_location="cpu", weights_only=False)
    model = {
        k: (v.half() if torch.is_floating_point(v) else v)
        for k, v in state["model"].items()
    }
    torch.save({"model": model, "variant": args.variant}, args.dst)

    n_params = sum(v.numel() for v in model.values())
    with open(args.dst, "rb") as handle:
        sha = hashlib.sha256(handle.read()).hexdigest()
    print(f"{args.dst}: {n_params / 1e6:.1f}M parameters, sha256 {sha}")


if __name__ == "__main__":
    main()
