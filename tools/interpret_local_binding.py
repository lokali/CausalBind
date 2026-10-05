#!/usr/bin/env python3
"""Local-binding case study for the AtomAttn CausalBind checkpoint.

For each requested DUD-E target, this script:
  1. encodes the aligned crystal ligand and binding pocket;
  2. exports final-layer concept-to-atom attention;
  3. identifies concepts that cover observed heavy-atom contacts (<= 4 A);
  4. ablates those concepts and compares the final pocket-ligand score with
     matched random concept ablations; and
  5. writes a JSON report plus a compact figure.

The contact-based concept selection is performed before looking at the
ablation response. Attention is treated as model attribution, not as proof of
a physical causal mechanism.
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import lmdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from rdkit import Chem
from unicore import checkpoint_utils, options, tasks, utils


def add_analysis_args(parser):
    parser.add_argument(
        "--case-targets",
        default="egfr,adrb2,cdk2,hivpr,esr1,src,fa10,thrb",
        help="Comma-separated DUD-E target directory names.",
    )
    parser.add_argument("--case-output", required=True)
    parser.add_argument("--contact-cutoff", type=float, default=4.0)
    parser.add_argument("--local-group-size", type=int, default=4)
    parser.add_argument("--random-controls", type=int, default=256)
    parser.add_argument("--random-atom-controls", type=int, default=64)
    parser.add_argument("--local-ligand-atoms", type=int, default=4)
    parser.add_argument("--local-pocket-atoms", type=int, default=8)
    parser.add_argument("--analysis-seed", type=int, default=2026)
    return parser


def read_lmdb_record(path):
    env = lmdb.open(
        str(path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    try:
        value = env.begin().get(b"0")
        if value is None:
            raise ValueError(f"No key 0 in {path}")
        return pickle.loads(value)
    finally:
        env.close()


def write_lmdb_record(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    env = lmdb.open(
        str(path),
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        map_size=1 << 30,
    )
    try:
        with env.begin(write=True) as txn:
            txn.put(b"0", pickle.dumps(record))
    finally:
        env.close()


def pocket_element(atom_name):
    atom_name = atom_name.strip()
    if not atom_name:
        return ""
    if atom_name[0].isdigit() and len(atom_name) > 1:
        return atom_name[1]
    return atom_name[0]


def load_crystal_ligand(path):
    mol = Chem.MolFromMol2File(str(path), removeHs=False, sanitize=True)
    if mol is None:
        mol = Chem.MolFromMol2File(str(path), removeHs=False, sanitize=False)
    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError(f"RDKit could not parse crystal ligand: {path}")

    coords_all = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)
    symbols_all = np.asarray([atom.GetSymbol() for atom in mol.GetAtoms()])
    heavy_indices = np.flatnonzero(symbols_all != "H")
    old_to_heavy = {int(old): new for new, old in enumerate(heavy_indices)}
    bonds = []
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if begin in old_to_heavy and end in old_to_heavy:
            bonds.append((old_to_heavy[begin], old_to_heavy[end]))

    heavy_symbols = symbols_all[heavy_indices].tolist()
    heavy_coords = coords_all[heavy_indices]
    record = {
        "atoms": symbols_all.tolist(),
        "coordinates": [coords_all],
        "smi": Chem.MolToSmiles(Chem.RemoveHs(mol)),
        "mol": mol,
        "label": 1,
    }
    return record, heavy_symbols, heavy_coords, bonds


def load_pocket_geometry(path):
    record = read_lmdb_record(path)
    atom_names_all = np.asarray(record["pocket_atoms"])
    elements_all = np.asarray([pocket_element(x) for x in atom_names_all])
    coords_all = np.asarray(record["pocket_coordinates"], dtype=np.float32)
    keep = elements_all != "H"
    return (
        record,
        atom_names_all[keep].tolist(),
        elements_all[keep].tolist(),
        coords_all[keep],
    )


def parse_pdb_atoms(path):
    atoms = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            try:
                coord = np.array(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ],
                    dtype=np.float32,
                )
            except ValueError:
                continue
            atoms.append(
                {
                    "coord": coord,
                    "atom_name": line[12:16].strip(),
                    "resname": line[17:20].strip(),
                    "chain": line[21].strip() or "_",
                    "resseq": line[22:26].strip(),
                    "icode": line[26].strip(),
                }
            )
    return atoms


def match_pocket_to_residues(pocket_coords, pdb_atoms):
    pdb_coords = np.stack([atom["coord"] for atom in pdb_atoms])
    labels = []
    atom_labels = []
    errors = []
    for coord in pocket_coords:
        distances = np.linalg.norm(pdb_coords - coord[None, :], axis=1)
        index = int(np.argmin(distances))
        atom = pdb_atoms[index]
        residue = (
            f"{atom['resname']} {atom['chain']}:{atom['resseq']}"
            f"{atom['icode']}"
        )
        labels.append(residue)
        atom_labels.append(f"{residue}/{atom['atom_name']}")
        errors.append(float(distances[index]))
    return labels, atom_labels, errors


def single_sample(dataset):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=dataset.collater,
    )
    return next(iter(loader))


def atom_attention(result):
    # Last Perceiver layer; average heads. Exclude BOS and EOS, then
    # renormalize over real atoms independently for each concept.
    attention = result["attention_layers"][-1, 0].float().mean(dim=0)
    valid_length = int((~result["padding_mask"][0]).sum().item())
    attention = attention[:, 1 : valid_length - 1]
    attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return attention.detach().cpu().numpy()


def batch_ablation_embeddings(model, concepts, branch, groups):
    batch = concepts.expand(len(groups), -1, -1).clone()
    for row, group in enumerate(groups):
        batch[row, list(group), :] = 0
    return model.embedding_from_concepts(batch, branch)


def batch_token_occlusion_embeddings(
    model,
    encoder_output,
    base_padding_mask,
    branch,
    atom_groups,
    batch_size=16,
):
    outputs = []
    for start in range(0, len(atom_groups), batch_size):
        groups = atom_groups[start : start + batch_size]
        token_mask = base_padding_mask.expand(len(groups), -1).clone()
        for row, group in enumerate(groups):
            # Atom index 0 corresponds to token index 1 after BOS insertion.
            token_mask[row, np.asarray(group, dtype=np.int64) + 1] = True
        encoder_batch = encoder_output.expand(len(groups), -1, -1)
        outputs.append(
            model.embedding_from_encoder_tokens(
                encoder_batch, token_mask, branch
            )
        )
    return torch.cat(outputs, dim=0)


def percentile_of_score(values, observed):
    values = np.asarray(values)
    return float(100.0 * (np.sum(values <= observed) + 0.5) / (len(values) + 1.0))


def standardize(values):
    values = np.asarray(values, dtype=np.float64)
    std = values.std()
    if std < 1e-12:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def choose_local_concepts(contact_overlap, group_size):
    # This selection uses contact coverage only; no binding-score response is
    # consulted. Tiny deterministic index offsets make tie-breaking stable.
    mol_coverage = contact_overlap.sum(axis=1)
    pocket_coverage = contact_overlap.sum(axis=0)
    mol_order = np.argsort(
        mol_coverage + np.arange(len(mol_coverage)) * 1e-15
    )[::-1]
    pocket_order = np.argsort(
        pocket_coverage + np.arange(len(pocket_coverage)) * 1e-15
    )[::-1]
    return mol_order[:group_size], pocket_order[:group_size]


def analyze_target(model, task, data_root, target, output_root, args):
    target_dir = data_root / "DUD-E" / target
    ligand_path = target_dir / "crystal_ligand.mol2"
    pocket_path = target_dir / "pocket.lmdb"
    receptor_path = target_dir / "receptor.pdb"
    for required in (ligand_path, pocket_path, receptor_path):
        if not required.exists():
            raise FileNotFoundError(required)

    target_output = output_root / target
    target_output.mkdir(parents=True, exist_ok=True)
    ligand_record, ligand_elements, ligand_coords, bonds = load_crystal_ligand(
        ligand_path
    )
    ligand_lmdb = target_output / "crystal_ligand_input.lmdb"
    write_lmdb_record(ligand_lmdb, ligand_record)
    _, pocket_atom_names, pocket_elements, pocket_coords = load_pocket_geometry(
        pocket_path
    )
    pdb_atoms = parse_pdb_atoms(receptor_path)
    residue_labels, pocket_atom_labels, match_errors = match_pocket_to_residues(
        pocket_coords, pdb_atoms
    )

    mol_dataset = task.load_mols_dataset(
        str(ligand_lmdb), "atoms", "coordinates"
    )
    pocket_dataset = task.load_pockets_dataset(str(pocket_path))
    mol_sample = utils.move_to_cuda(single_sample(mol_dataset))
    pocket_sample = utils.move_to_cuda(single_sample(pocket_dataset))

    with torch.inference_mode():
        mol_result = model.mol_forward_with_attention(**mol_sample["net_input"])
        pocket_result = model.pocket_forward_with_attention(
            **pocket_sample["net_input"]
        )

    mol_attention = atom_attention(mol_result)
    pocket_attention = atom_attention(pocket_result)
    if mol_attention.shape[1] != len(ligand_coords):
        raise ValueError(
            f"Ligand attention/coordinate mismatch: "
            f"{mol_attention.shape[1]} vs {len(ligand_coords)}"
        )
    if pocket_attention.shape[1] != len(pocket_coords):
        raise ValueError(
            f"Pocket attention/coordinate mismatch: "
            f"{pocket_attention.shape[1]} vs {len(pocket_coords)}"
        )

    distances = np.linalg.norm(
        ligand_coords[:, None, :] - pocket_coords[None, :, :], axis=-1
    )
    contact_matrix = distances <= args.contact_cutoff
    if not contact_matrix.any():
        raise ValueError(f"No <= {args.contact_cutoff} A contact for {target}")

    contact_overlap = mol_attention @ contact_matrix.astype(np.float64)
    contact_overlap = contact_overlap @ pocket_attention.T
    local_mol, local_pocket = choose_local_concepts(
        contact_overlap, args.local_group_size
    )
    local_pair_flat = int(np.argmax(contact_overlap))
    local_pair = np.unravel_index(local_pair_flat, contact_overlap.shape)

    base_score = float(
        (
            pocket_result["embedding"].float()
            @ mol_result["embedding"].float().T
        )[0, 0].item()
    )
    with torch.inference_mode():
        local_mol_embedding = batch_ablation_embeddings(
            model,
            mol_result["concepts"],
            "mol",
            [tuple(local_mol)],
        )
        local_pocket_embedding = batch_ablation_embeddings(
            model,
            pocket_result["concepts"],
            "pocket",
            [tuple(local_pocket)],
        )
        local_score = float(
            (local_pocket_embedding.float() @ local_mol_embedding.float().T)[
                0, 0
            ].item()
        )
        pair_mol_embedding = batch_ablation_embeddings(
            model,
            mol_result["concepts"],
            "mol",
            [(int(local_pair[0]),)],
        )
        pair_pocket_embedding = batch_ablation_embeddings(
            model,
            pocket_result["concepts"],
            "pocket",
            [(int(local_pair[1]),)],
        )
        pair_score = float(
            (pair_pocket_embedding.float() @ pair_mol_embedding.float().T)[
                0, 0
            ].item()
        )

        rng = np.random.default_rng(args.analysis_seed)
        num_concepts = mol_attention.shape[0]
        random_mol_groups = [
            tuple(
                sorted(
                    rng.choice(
                        num_concepts,
                        size=args.local_group_size,
                        replace=False,
                    ).tolist()
                )
            )
            for _ in range(args.random_controls)
        ]
        random_pocket_groups = [
            tuple(
                sorted(
                    rng.choice(
                        num_concepts,
                        size=args.local_group_size,
                        replace=False,
                    ).tolist()
                )
            )
            for _ in range(args.random_controls)
        ]
        random_mol_embeddings = batch_ablation_embeddings(
            model, mol_result["concepts"], "mol", random_mol_groups
        )
        random_pocket_embeddings = batch_ablation_embeddings(
            model,
            pocket_result["concepts"],
            "pocket",
            random_pocket_groups,
        )
        random_scores = (
            random_pocket_embeddings.float()
            * random_mol_embeddings.float()
        ).sum(dim=1)
        random_scores = random_scores.detach().cpu().numpy()

    selected_drop = base_score - local_score
    pair_drop = base_score - pair_score
    random_drops = base_score - random_scores
    drop_percentile = percentile_of_score(random_drops, selected_drop)

    selected_mol_saliency = mol_attention[local_mol].mean(axis=0)
    selected_pocket_saliency = pocket_attention[local_pocket].mean(axis=0)
    ligand_contact_atoms = contact_matrix.any(axis=1)
    pocket_contact_atoms = contact_matrix.any(axis=0)
    mol_contact_mass = float(selected_mol_saliency[ligand_contact_atoms].sum())
    pocket_contact_mass = float(
        selected_pocket_saliency[pocket_contact_atoms].sum()
    )
    mol_uniform_mass = float(ligand_contact_atoms.mean())
    pocket_uniform_mass = float(pocket_contact_atoms.mean())
    uniform_pair_contact = float(contact_matrix.mean())
    pair_contact_probability = float(contact_overlap[local_pair])

    # Atom-level occlusion uses actual contact atoms ranked by attention from
    # the independently contact-selected concept groups. It therefore tests a
    # more literal local-to-global path than concept deletion alone.
    ligand_contact_order = np.flatnonzero(ligand_contact_atoms)
    ligand_contact_order = ligand_contact_order[
        np.argsort(selected_mol_saliency[ligand_contact_order])[::-1]
    ]
    pocket_contact_order = np.flatnonzero(pocket_contact_atoms)
    pocket_contact_order = pocket_contact_order[
        np.argsort(selected_pocket_saliency[pocket_contact_order])[::-1]
    ]
    local_ligand_atom_count = min(
        args.local_ligand_atoms, len(ligand_contact_order)
    )
    local_pocket_atom_count = min(
        args.local_pocket_atoms, len(pocket_contact_order)
    )
    local_ligand_atoms = ligand_contact_order[:local_ligand_atom_count]
    local_pocket_atoms = pocket_contact_order[:local_pocket_atom_count]

    rng_atoms = np.random.default_rng(args.analysis_seed + 17)
    random_ligand_atom_groups = [
        tuple(
            sorted(
                rng_atoms.choice(
                    len(ligand_coords),
                    size=local_ligand_atom_count,
                    replace=False,
                ).tolist()
            )
        )
        for _ in range(args.random_atom_controls)
    ]
    random_pocket_atom_groups = [
        tuple(
            sorted(
                rng_atoms.choice(
                    len(pocket_coords),
                    size=local_pocket_atom_count,
                    replace=False,
                ).tolist()
            )
        )
        for _ in range(args.random_atom_controls)
    ]
    with torch.inference_mode():
        local_mol_atom_embedding = batch_token_occlusion_embeddings(
            model,
            mol_result["encoder_output"],
            mol_result["padding_mask"],
            "mol",
            [tuple(int(x) for x in local_ligand_atoms)],
        )
        local_pocket_atom_embedding = batch_token_occlusion_embeddings(
            model,
            pocket_result["encoder_output"],
            pocket_result["padding_mask"],
            "pocket",
            [tuple(int(x) for x in local_pocket_atoms)],
        )
        local_atom_score = float(
            (
                local_pocket_atom_embedding.float()
                @ local_mol_atom_embedding.float().T
            )[0, 0].item()
        )
        random_mol_atom_embeddings = batch_token_occlusion_embeddings(
            model,
            mol_result["encoder_output"],
            mol_result["padding_mask"],
            "mol",
            random_ligand_atom_groups,
        )
        random_pocket_atom_embeddings = batch_token_occlusion_embeddings(
            model,
            pocket_result["encoder_output"],
            pocket_result["padding_mask"],
            "pocket",
            random_pocket_atom_groups,
        )
        random_atom_scores = (
            random_pocket_atom_embeddings.float()
            * random_mol_atom_embeddings.float()
        ).sum(dim=1)
        random_atom_scores = random_atom_scores.detach().cpu().numpy()
    local_atom_drop = base_score - local_atom_score
    random_atom_drops = base_score - random_atom_scores
    atom_drop_percentile = percentile_of_score(
        random_atom_drops, local_atom_drop
    )

    residue_saliency = {}
    residue_min_distance = {}
    for index, residue in enumerate(residue_labels):
        residue_saliency[residue] = residue_saliency.get(residue, 0.0) + float(
            selected_pocket_saliency[index]
        )
        residue_min_distance[residue] = min(
            residue_min_distance.get(residue, np.inf),
            float(distances[:, index].min()),
        )
    top_residues = sorted(
        (
            {
                "residue": residue,
                "attention": saliency,
                "min_distance_A": residue_min_distance[residue],
            }
            for residue, saliency in residue_saliency.items()
        ),
        key=lambda item: (
            item["min_distance_A"] > args.contact_cutoff,
            -item["attention"],
        ),
    )[:10]

    contact_pairs = np.argwhere(contact_matrix)
    closest_pairs = sorted(
        (
            {
                "ligand_atom_index": int(i),
                "ligand_element": ligand_elements[i],
                "pocket_atom_index": int(j),
                "pocket_atom": pocket_atom_labels[j],
                "distance_A": float(distances[i, j]),
            }
            for i, j in contact_pairs
        ),
        key=lambda item: item["distance_A"],
    )[:20]

    report = {
        "target": target,
        "checkpoint": str(args.path),
        "selection_rule": (
            f"Top {args.local_group_size} ligand and pocket concepts by "
            f"coverage of observed <= {args.contact_cutoff:.1f} A heavy-atom "
            "contacts; binding-score response was not used for selection."
        ),
        "contact_cutoff_A": args.contact_cutoff,
        "num_ligand_heavy_atoms": len(ligand_coords),
        "num_pocket_heavy_atoms": len(pocket_coords),
        "num_heavy_atom_contacts": int(contact_matrix.sum()),
        "num_contacting_ligand_atoms": int(ligand_contact_atoms.sum()),
        "num_contacting_pocket_atoms": int(pocket_contact_atoms.sum()),
        "local_molecule_concepts": [int(x) for x in local_mol],
        "local_pocket_concepts": [int(x) for x in local_pocket],
        "top_contact_concept_pair": [
            int(local_pair[0]),
            int(local_pair[1]),
        ],
        "baseline_binding_score": base_score,
        "local_group_ablated_score": local_score,
        "local_group_score_drop": selected_drop,
        "local_group_relative_drop_percent": (
            100.0 * selected_drop / abs(base_score)
            if abs(base_score) > 1e-12
            else None
        ),
        "top_pair_ablated_score": pair_score,
        "top_pair_score_drop": pair_drop,
        "random_group_score_drop_mean": float(random_drops.mean()),
        "random_group_score_drop_std": float(random_drops.std()),
        "local_drop_percentile_vs_random": drop_percentile,
        "occluded_local_ligand_atom_indices": [
            int(x) for x in local_ligand_atoms
        ],
        "occluded_local_pocket_atom_indices": [
            int(x) for x in local_pocket_atoms
        ],
        "local_contact_atom_occluded_score": local_atom_score,
        "local_contact_atom_score_drop": local_atom_drop,
        "local_contact_atom_relative_drop_percent": (
            100.0 * local_atom_drop / abs(base_score)
            if abs(base_score) > 1e-12
            else None
        ),
        "random_atom_score_drop_mean": float(random_atom_drops.mean()),
        "random_atom_score_drop_std": float(random_atom_drops.std()),
        "local_atom_drop_percentile_vs_random": atom_drop_percentile,
        "ligand_contact_attention_mass": mol_contact_mass,
        "ligand_contact_atom_fraction": mol_uniform_mass,
        "ligand_contact_attention_enrichment": (
            mol_contact_mass / mol_uniform_mass
        ),
        "pocket_contact_attention_mass": pocket_contact_mass,
        "pocket_contact_atom_fraction": pocket_uniform_mass,
        "pocket_contact_attention_enrichment": (
            pocket_contact_mass / pocket_uniform_mass
        ),
        "top_pair_contact_probability": pair_contact_probability,
        "uniform_pair_contact_probability": uniform_pair_contact,
        "top_pair_contact_enrichment": (
            pair_contact_probability / uniform_pair_contact
        ),
        "max_pdb_coordinate_match_error_A": float(max(match_errors)),
        "top_contact_residues": top_residues,
        "closest_contacts": closest_pairs,
        "interpretation_caveat": (
            "Attention and concept ablation provide model-level attribution. "
            "The geometric contacts are observed in the aligned crystal "
            "structure; this analysis does not by itself prove a physical "
            "causal mechanism."
        ),
    }
    with open(target_output / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    make_figure(
        target_output,
        report,
        ligand_coords,
        pocket_coords,
        ligand_elements,
        bonds,
        contact_matrix,
        distances,
        residue_labels,
        selected_mol_saliency,
        selected_pocket_saliency,
        contact_overlap,
        local_mol,
        local_pocket,
        pair_drop,
        selected_drop,
        random_drops,
        local_atom_drop,
        random_atom_drops,
    )
    return report


def pca_project(ligand_coords, pocket_coords, include_mask):
    selected_pocket = pocket_coords[include_mask]
    all_coords = np.concatenate([ligand_coords, selected_pocket], axis=0)
    center = ligand_coords.mean(axis=0)
    _, _, vt = np.linalg.svd(all_coords - center, full_matrices=False)
    return (
        (ligand_coords - center) @ vt[:2].T,
        (pocket_coords - center) @ vt[:2].T,
    )


def make_figure(
    output_dir,
    report,
    ligand_coords,
    pocket_coords,
    ligand_elements,
    bonds,
    contact_matrix,
    distances,
    residue_labels,
    mol_saliency,
    pocket_saliency,
    contact_overlap,
    local_mol,
    local_pocket,
    pair_drop,
    selected_drop,
    random_drops,
    local_atom_drop,
    random_atom_drops,
):
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "font.family": "DejaVu Sans",
        }
    )
    fig = plt.figure(figsize=(14.5, 4.8), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 0.9])
    ax_local = fig.add_subplot(grid[0, 0])
    ax_heat = fig.add_subplot(grid[0, 1])
    ax_drop = fig.add_subplot(grid[0, 2])

    contact_pocket = contact_matrix.any(axis=0)
    # Keep the local panel legible: show the most salient contacting pocket
    # atoms, the closest-contact endpoints, and every atom used in occlusion.
    contacting_indices = np.flatnonzero(contact_pocket)
    salient_contact_indices = contacting_indices[
        np.argsort(pocket_saliency[contacting_indices])[::-1][:24]
    ]
    closest_pocket_indices = np.argsort(distances.min(axis=0))[:12]
    occluded_pocket_indices = np.asarray(
        report["occluded_local_pocket_atom_indices"], dtype=np.int64
    )
    shown_indices = np.unique(
        np.concatenate(
            [
                salient_contact_indices,
                closest_pocket_indices,
                occluded_pocket_indices,
            ]
        )
    )
    display_pocket = np.zeros(len(pocket_coords), dtype=bool)
    display_pocket[shown_indices] = True
    ligand_xy, pocket_xy = pca_project(
        ligand_coords, pocket_coords, display_pocket
    )

    for begin, end in bonds:
        ax_local.plot(
            ligand_xy[[begin, end], 0],
            ligand_xy[[begin, end], 1],
            color="#333333",
            linewidth=1.4,
            zorder=2,
        )
    contact_pairs = np.argwhere(contact_matrix)
    # Show at most the 20 closest contacts to keep the panel readable.
    contact_pairs = sorted(
        contact_pairs,
        key=lambda pair: distances[pair[0], pair[1]],
    )[:20]
    for ligand_index, pocket_index in contact_pairs:
        ax_local.plot(
            [ligand_xy[ligand_index, 0], pocket_xy[pocket_index, 0]],
            [ligand_xy[ligand_index, 1], pocket_xy[pocket_index, 1]],
            color="#55a6a6",
            linestyle="--",
            linewidth=0.7,
            alpha=0.55,
            zorder=1,
        )

    ligand_norm = Normalize(
        vmin=float(mol_saliency.min()),
        vmax=float(mol_saliency.max() + 1e-12),
    )
    pocket_norm = Normalize(
        vmin=float(pocket_saliency.min()),
        vmax=float(pocket_saliency.max() + 1e-12),
    )
    ligand_scatter = ax_local.scatter(
        ligand_xy[:, 0],
        ligand_xy[:, 1],
        c=mol_saliency,
        cmap="YlOrRd",
        norm=ligand_norm,
        s=75,
        edgecolors="#222222",
        linewidths=0.6,
        label="Crystal ligand",
        zorder=4,
    )
    pocket_scatter = ax_local.scatter(
        pocket_xy[shown_indices, 0],
        pocket_xy[shown_indices, 1],
        c=pocket_saliency[shown_indices],
        cmap="Blues",
        norm=pocket_norm,
        s=55,
        marker="s",
        edgecolors="#24405c",
        linewidths=0.4,
        label="Pocket atoms",
        zorder=3,
    )
    occluded_ligand_indices = np.asarray(
        report["occluded_local_ligand_atom_indices"], dtype=np.int64
    )
    ax_local.scatter(
        ligand_xy[occluded_ligand_indices, 0],
        ligand_xy[occluded_ligand_indices, 1],
        s=120,
        facecolors="none",
        edgecolors="#00a6b2",
        linewidths=1.5,
        zorder=6,
        label="Occluded local atoms",
    )
    ax_local.scatter(
        pocket_xy[occluded_pocket_indices, 0],
        pocket_xy[occluded_pocket_indices, 1],
        s=105,
        marker="s",
        facecolors="none",
        edgecolors="#00a6b2",
        linewidths=1.5,
        zorder=6,
    )
    for index, element in enumerate(ligand_elements):
        ax_local.text(
            ligand_xy[index, 0],
            ligand_xy[index, 1],
            element,
            ha="center",
            va="center",
            fontsize=6.5,
            zorder=5,
        )

    residue_scores = {}
    residue_centers = {}
    for index in shown_indices:
        residue = residue_labels[index]
        residue_scores[residue] = residue_scores.get(residue, 0.0) + float(
            pocket_saliency[index]
        )
        residue_centers.setdefault(residue, []).append(pocket_xy[index])
    for residue, _ in sorted(
        residue_scores.items(), key=lambda item: item[1], reverse=True
    )[:6]:
        center = np.mean(residue_centers[residue], axis=0)
        ax_local.annotate(
            residue,
            xy=center,
            xytext=(3, 4),
            textcoords="offset points",
            fontsize=6.8,
            color="#173b62",
        )
    ax_local.set_title(
        f"A  {report['target'].upper()} local contacts (≤4 Å; 2D projection)",
        loc="left",
        fontweight="bold",
    )
    ax_local.set_xlabel("PCA axis 1 (Å)")
    ax_local.set_ylabel("PCA axis 2 (Å)")
    ax_local.set_aspect("equal", adjustable="datalim")
    ax_local.legend(loc="best", fontsize=7, frameon=False)
    ax_local.text(
        0.02,
        0.02,
        "Color intensity: within-branch concept attention",
        transform=ax_local.transAxes,
        fontsize=7,
        color="#444444",
    )

    image = ax_heat.imshow(
        contact_overlap,
        origin="lower",
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
    )
    ax_heat.scatter(
        local_pocket,
        local_mol,
        marker="s",
        facecolors="none",
        edgecolors="#55e6ff",
        s=55,
        linewidths=1.0,
        label="Contact-selected concepts",
    )
    pair = report["top_contact_concept_pair"]
    ax_heat.scatter(
        [pair[1]],
        [pair[0]],
        marker="x",
        c="white",
        s=70,
        linewidths=1.6,
        label=f"Top pair ({pair[0]}, {pair[1]})",
    )
    ax_heat.set_title(
        "B  Concept-pair contact overlap",
        loc="left",
        fontweight="bold",
    )
    ax_heat.set_xlabel("Pocket concept")
    ax_heat.set_ylabel("Ligand concept")
    ax_heat.legend(loc="upper right", fontsize=7, frameon=True)
    heat_cbar = fig.colorbar(image, ax=ax_heat, fraction=0.046, pad=0.04)
    heat_cbar.set_label("Attention-weighted contact probability", fontsize=7)

    random_atom_mean = float(np.mean(random_atom_drops))
    random_atom_std = float(np.std(random_atom_drops))
    bars = ax_drop.bar(
        [
            "Local\ncontacts",
            "Random\natoms",
            "Local\nconcepts",
        ],
        [local_atom_drop, random_atom_mean, selected_drop],
        yerr=[0.0, random_atom_std, 0.0],
        color=["#d95f59", "#a6a6a6", "#7c4d9e"],
        edgecolor="#333333",
        linewidth=0.6,
        capsize=4,
    )
    ax_drop.axhline(0.0, color="#222222", linewidth=0.8)
    ax_drop.set_ylabel("Decrease in final binding score")
    ax_drop.set_title(
        "C  Local-to-global occlusion",
        loc="left",
        fontweight="bold",
    )
    for bar, value in zip(
        bars, [local_atom_drop, random_atom_mean, selected_drop]
    ):
        offset = 0.02 * max(
            np.max(np.abs(random_atom_drops)),
            abs(local_atom_drop),
            abs(selected_drop),
            1e-6,
        )
        ax_drop.text(
            bar.get_x() + bar.get_width() / 2,
            value + (offset if value >= 0 else -offset),
            f"{value:.3g}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )
    ax_drop.text(
        0.98,
        0.98,
        (
            f"Contact atoms: "
            f"{report['local_atom_drop_percentile_vs_random']:.1f}th pct.\n"
            f"Contact concepts: "
            f"{report['local_drop_percentile_vs_random']:.1f}th pct.\n"
            "Matched random controls"
        ),
        transform=ax_drop.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#bbbbbb",
            "alpha": 0.9,
        },
    )

    fig.suptitle(
        "CausalBind local-to-global attribution in an aligned DUD-E "
        "crystal complex",
        fontsize=12.5,
        fontweight="bold",
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"local_binding.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("This analysis requires a CUDA node.")
    torch.cuda.set_device(args.device_id)
    np.random.seed(args.analysis_seed)
    torch.manual_seed(args.analysis_seed)

    print(f"Loading checkpoint: {args.path}", flush=True)
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    if unexpected:
        print(f"Unexpected checkpoint keys: {unexpected[:10]}", flush=True)
    if missing:
        print(f"Missing checkpoint keys: {missing[:10]}", flush=True)
    if args.fp16:
        model.half()
    model.cuda().eval()

    output_root = Path(args.case_output)
    output_root.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data)
    reports = []
    failures = {}
    targets = [x.strip() for x in args.case_targets.split(",") if x.strip()]
    for target in targets:
        print(f"Analyzing DUD-E target: {target}", flush=True)
        try:
            report = analyze_target(
                model, task, data_root, target, output_root, args
            )
            reports.append(report)
            print(
                f"  score={report['baseline_binding_score']:.6g}, "
                f"drop={report['local_group_score_drop']:.6g}, "
                f"random percentile="
                f"{report['local_drop_percentile_vs_random']:.1f}",
                flush=True,
            )
        except Exception as error:
            failures[target] = f"{type(error).__name__}: {error}"
            print(f"  FAILED: {failures[target]}", flush=True)

    summary = {
        "reports": reports,
        "failures": failures,
        "recommended_case": None,
    }
    if reports:
        # Recommend the strongest functional response among independently
        # contact-selected concepts. The per-target reports remain available,
        # making this case-level selection transparent.
        recommended = max(
            reports,
            key=lambda report: (
                report["local_atom_drop_percentile_vs_random"],
                report["local_drop_percentile_vs_random"],
                report["local_group_score_drop"],
            ),
        )
        summary["recommended_case"] = recommended["target"]
    with open(
        output_root / "summary.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
    print(
        f"Recommended illustrative case: {summary['recommended_case']}",
        flush=True,
    )


def cli_main():
    parser = options.get_validation_parser()
    parser.add_argument(
        "--test-task",
        type=str,
        default="DUDE",
        choices=["DUDE"],
    )
    add_analysis_args(parser)
    options.add_model_args(parser)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
