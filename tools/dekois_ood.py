#!/usr/bin/env python3
"""Out-of-distribution evaluation on DEKOIS 2.0 (paper App. A6.3).

Two complementary settings, both defined from the training labels alone and
fixed before any model is evaluated:

  * Target-level OOD: DEKOIS targets whose UniProt IDs are absent from all
    training sources. The training task already removes every DEKOIS UniProt
    ID from the ChEMBL/BindingDB assays, but not from the PDBbind pairs, so a
    target qualifies when it also does not occur in PDBbind.
  * Scaffold-level OOD: within those targets, the test pairs whose achiral
    Bemis-Murcko scaffold is absent from every raw training ligand; targets
    with fewer than --min-unseen-actives such actives are skipped.

Usage:
  # 1) build the target-level OOD subset (symlinks into the full DEKOIS 2.0x)
  python tools/dekois_ood.py make-subset --out test_datasets/DEKOIS_OOD

  # 2) evaluate a checkpoint on it (writes <results>/DEKOIS/<target>/saved_*.npy)
  TEST_DATA_ROOT=test_datasets/DEKOIS_OOD bash scripts/test.sh sp DEKOIS ckpt.pt results/sp_dekois

  # 3) scaffold-level metrics from the saved predictions
  python tools/dekois_ood.py scaffold --subset test_datasets/DEKOIS_OOD \
      --results CausalBind-SP=results/sp_dekois/DEKOIS
"""

import argparse
import json
import multiprocessing as mp
import os
import pickle
from pathlib import Path

import lmdb
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Scoring.Scoring import CalcAUC, CalcBEDROC, CalcEnrichment

REPO_DIR = Path(__file__).resolve().parents[1]
RDLogger.DisableLog("rdApp.*")


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def load_lmdb_records(path):
    env = lmdb.open(str(path), readonly=True, lock=False, subdir=False)
    with env.begin() as txn:
        records = [pickle.loads(value) for _, value in txn.cursor()]
    env.close()
    return records


def scaffold_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) or None


def metrics(labels, predictions):
    ranked = np.column_stack([predictions, labels])
    ranked = ranked[np.argsort(ranked[:, 0])[::-1]]
    enrichment = CalcEnrichment(ranked, 1, [0.005, 0.01, 0.02, 0.05])
    return {
        "AUROC": float(CalcAUC(ranked, 1)),
        "BEDROC": float(CalcBEDROC(ranked, 1, 80.5)),
        "EF0.5": float(enrichment[0]),
        "EF1": float(enrichment[1]),
        "EF2": float(enrichment[2]),
        "EF5": float(enrichment[3]),
    }


def make_subset(args):
    pdbbind = load_json(args.data_dir / "train_label_pdbbind_seq.json")
    pdbbind_ids = {row["uniprot"] for row in pdbbind}
    dekois = load_json(args.test_dir / "dekois.json")  # [uniprot, pdb_id, TARGET]

    selected = [row for row in dekois if row[0] not in pdbbind_ids]
    out_dir = args.out / "DEKOIS_2.0x"
    out_dir.mkdir(parents=True, exist_ok=True)
    for uniprot, pdb_id, target in selected:
        src = (args.dekois_dir / target.lower()).resolve()
        if not src.is_dir():
            raise FileNotFoundError(f"missing DEKOIS target directory: {src}")
        link = out_dir / target.lower()
        if not link.exists():
            os.symlink(src, link)
        print(f"{target:12s} {uniprot:10s} {pdb_id}")
    json_link = args.out / "dekois.json"
    if not json_link.exists():
        os.symlink((args.test_dir / "dekois.json").resolve(), json_link)
    print(f"{len(selected)} target-level OOD targets -> {args.out}")


def scaffold(args):
    train_rows = load_json(args.data_dir / "train_label_pdbbind_seq.json") + load_json(
        args.data_dir / "train_label_blend_seq_full.json"
    )
    train_smiles = {lig["smi"] for row in train_rows for lig in row["ligands"] if lig.get("smi")}
    del train_rows
    with mp.Pool(args.workers) as pool:
        train_scaffolds = {
            s for s in pool.imap_unordered(scaffold_from_smiles, train_smiles, chunksize=512) if s
        }
    print(f"{len(train_smiles)} unique training molecules, {len(train_scaffolds)} scaffolds")

    data_root = args.subset / "DEKOIS_2.0x"
    masks, all_labels, counts = {}, {}, {}
    for target_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        target = target_dir.name
        records = load_lmdb_records(target_dir / f"{target}_lig.lmdb")
        labels = np.asarray([r["label"] for r in records])
        mask = np.asarray(
            [(s is not None and s not in train_scaffolds)
             for s in (scaffold_from_smiles(r["smi"]) for r in records)]
        )
        masks[target], all_labels[target] = mask, labels
        counts[target] = {"unseen_pairs": int(mask.sum()), "unseen_actives": int(labels[mask].sum())}
    eligible = sorted(t for t, c in counts.items() if c["unseen_actives"] >= args.min_unseen_actives)
    n_pairs = sum(counts[t]["unseen_pairs"] for t in eligible)
    n_act = sum(counts[t]["unseen_actives"] for t in eligible)
    print(f"{len(eligible)} targets with >= {args.min_unseen_actives} scaffold-unseen actives "
          f"({n_pairs} pairs, {n_act} actives): {', '.join(eligible)}")

    results = {}
    for spec in args.results:
        name, result_dir = spec.split("=", 1)
        per_target = {}
        for target in eligible:
            tdir = Path(result_dir) / target
            labels = np.load(tdir / "saved_labels.npy")
            preds = np.load(tdir / "saved_preds.npy")
            if not np.array_equal(labels, all_labels[target]):
                raise RuntimeError(f"label order mismatch for {name}/{target}")
            per_target[target] = metrics(labels[masks[target]], preds[masks[target]])
        agg = {k: float(np.mean([v[k] for v in per_target.values()])) for k in next(iter(per_target.values()))}
        results[name] = {"aggregate": agg, "targets": per_target}
        print(f"{name:20s} AUROC={agg['AUROC']:.3f} BEDROC={agg['BEDROC']:.3f} "
              f"EF@1%={agg['EF1']:.2f} EF@5%={agg['EF5']:.2f}")
    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"eligible_targets": eligible, "target_counts": counts, "methods": results}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=REPO_DIR / "data")
    parser.add_argument("--test-dir", type=Path, default=REPO_DIR / "test_datasets")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("make-subset", help="build the target-level OOD subset")
    p.add_argument("--dekois-dir", type=Path, default=REPO_DIR / "test_datasets" / "DEKOIS_2.0x")
    p.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("scaffold", help="scaffold-level OOD metrics from saved predictions")
    p.add_argument("--subset", type=Path, required=True, help="directory created by make-subset")
    p.add_argument("--results", nargs="+", required=True, metavar="NAME=DIR",
                   help="<results_path>/DEKOIS directories written by scripts/test.sh")
    p.add_argument("--min-unseen-actives", type=int, default=5)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--json-out", type=Path)

    args = parser.parse_args()
    make_subset(args) if args.cmd == "make-subset" else scaffold(args)


if __name__ == "__main__":
    main()
