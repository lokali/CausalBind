# Data and Pretrained Backbones

CausalBind does not create or redistribute any dataset. All training and evaluation data are the public resources released with [LigUnity](https://github.com/IDEA-XL/LigUnity) and used by [HypSeek](https://github.com/jianhuiwemi/HypSeek), and remain subject to the terms set by their original providers. `scripts/download_data.sh` fetches everything that can be downloaded directly.

## Sources

| Asset | Source | License / terms | Used for |
|---|---|---|---|
| Uni-Mol molecule and pocket encoders (`mol_pre_no_h_220816.pt`, `pocket_pre_220816.pt`) | [Uni-Mol v0.1 release](https://github.com/deepmodeling/Uni-Mol/releases/tag/v0.1) | MIT | molecule / pocket backbones |
| ESM-2 35M (`esm2_t12_35M_UR50D`) | [Hugging Face](https://huggingface.co/facebook/esm2_t12_35M_UR50D) | MIT | protein-sequence backbone (frozen) |
| LigUnity training corpus (ChEMBL, BindingDB, PDBbind) | [figshare 10.6084/m9.figshare.27966819](https://doi.org/10.6084/m9.figshare.27966819) | CC BY-NC 4.0 | training |
| CASF validation set and benchmark index files | [LigUnity `test_datasets/`](https://github.com/IDEA-XL/LigUnity/tree/main/test_datasets) | CC BY-NC 4.0 | validation, target lists, FEP-overlap filters |
| DUD-E and LIT-PCBA (`dude.zip`, `pcba.zip`) | [LigUnity Google Drive](https://drive.google.com/drive/folders/1zW1MGpgunynFxTKXC2Q4RgWxZmg6CInV) | terms of the original benchmarks | evaluation |
| DEKOIS 2.0 | [figshare 10.6084/m9.figshare.27967422](https://doi.org/10.6084/m9.figshare.27967422) | terms of the original benchmark | out-of-distribution evaluation (paper App. A6.3) |
| FEP benchmarks (JACS-8, Merck-FEP-8) | [LigUnity `test_datasets/FEP`](https://github.com/IDEA-XL/LigUnity/tree/main/test_datasets/FEP) | CC BY-NC 4.0 | affinity ranking |

## Expected Layout

```
CausalBind/
├── pretrain/
│   ├── mol_pre_no_h_220816.pt
│   ├── pocket_pre_220816.pt
│   └── esm2_t12_35M_UR50D/
├── data/                                   # training corpus
│   ├── train_lig_all_blend.lmdb
│   ├── train_prot_all_blend.lmdb
│   ├── train_label_blend_seq_full.json     # from train_label.zip
│   ├── train_label_pdbbind_seq.json        # from train_label.zip
│   ├── mol_smi2idx_train_blend.json
│   ├── pocket_name2idx_train_blend.json
│   ├── valid_lig.lmdb, valid_prot.lmdb, valid_label_seq.json
│   ├── uniport40.clstr, uniport80.clstr
│   └── fep_assays.json, fep_repeat_ligands_can.json, fep_similar_ligands_0d5.json  # links to test_datasets/
└── test_datasets/
    ├── casf.lmdb, casf_label_seq.json
    ├── dude.json, PCBA.json, FEP.json, dekois.json
    ├── fep_assay_ids.json, fep_repeat_ligands_can.json, fep_similar_ligands_0d5.json
    ├── DUD-E/<target>/{mols.lmdb, pocket.lmdb, receptor.pdb, crystal_ligand.mol2, ...}   # 102 targets
    ├── lit_pcba/<TARGET>/{mols.lmdb, pockets.lmdb, ...}                                # 15 targets
    └── DEKOIS_2.0x/<target>/{<target>_lig.lmdb, <target>_pocket.lmdb, ...}             # 81 targets
```

The training task also reads `test_datasets/dude.json`, `PCBA.json`, and `dekois.json` to remove benchmark targets from the training assays, so the test index files are required for training as well.

Paths can be overridden with environment variables: `TEST_DATA_ROOT` (test benchmarks), `ESM2_PATH` (ESM-2 directory; if it does not exist, the model is loaded from Hugging Face), `UNIPROT_FASTA_DIR` (local FASTA cache, default `uniprot_fasta/`), and `UNIPROT_SEQ_CACHE` (optional JSON mapping UniProt IDs to sequences). Protein sequences of the test targets are fetched from UniProt on first use and cached in `uniprot_fasta/`; on machines without internet access, populate this directory beforehand.

## Train/Test Protocol

Evaluation follows a cross-dataset generalization protocol rather than a random split of the training pool. Following LigUnity and HypSeek, the training task removes ChEMBL/BindingDB assays that overlap the FEP benchmarks and every ChEMBL/BindingDB assay whose target UniProt ID appears in DUD-E, LIT-PCBA, or DEKOIS 2.0 (26,729 to 18,316 assays). The PDBbind structural pairs are not filtered by UniProt ID. CASF-2016 is used only for validation, and the paper reports the last checkpoint (epoch 50) of each run.
