#!/usr/bin/env bash
# Download the public assets used by CausalBind into the repository root.
# All datasets are third-party resources released by their original authors
# (see DATA.md for sources and licenses); this script only fetches them.
#
# Usage: bash scripts/download_data.sh [all|pretrain|train|test]
set -euo pipefail

part="${1:-all}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_DIR}"

fetch() {  # fetch <url> <output>
  [ -s "$2" ] && { echo "  exists: $2"; return; }
  echo "  downloading: $2"
  wget -nv -O "$2" "$1"
}

download_pretrain() {
  echo "[1/3] Pretrained backbones -> pretrain/"
  mkdir -p pretrain
  # Uni-Mol molecule and pocket encoders (Uni-Mol v0.1 release)
  fetch https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/mol_pre_no_h_220816.pt pretrain/mol_pre_no_h_220816.pt
  fetch https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/pocket_pre_220816.pt pretrain/pocket_pre_220816.pt
  # ESM-2 (35M) protein language model from Hugging Face
  python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("facebook/esm2_t12_35M_UR50D", local_dir="pretrain/esm2_t12_35M_UR50D")
EOF
}

download_train() {
  echo "[2/3] LigUnity training data (figshare 10.6084/m9.figshare.27966819) -> data/"
  mkdir -p data
  fetch https://ndownloader.figshare.com/files/50992173 data/valid_label_seq.json
  fetch https://ndownloader.figshare.com/files/50992176 data/valid_lig.lmdb
  fetch https://ndownloader.figshare.com/files/50992182 data/valid_prot.lmdb
  fetch https://ndownloader.figshare.com/files/50992179 data/train_label.zip
  fetch https://ndownloader.figshare.com/files/50992185 data/pocket_name2idx_train_blend.json
  fetch https://ndownloader.figshare.com/files/50992191 data/mol_smi2idx_train_blend.json
  fetch https://ndownloader.figshare.com/files/50992194 data/train_prot_all_blend.lmdb
  fetch https://ndownloader.figshare.com/files/50992728 data/train_lig_all_blend.lmdb
  fetch https://ndownloader.figshare.com/files/54239555 data/uniport80.clstr
  fetch https://ndownloader.figshare.com/files/54239558 data/uniport40.clstr
  (cd data && unzip -n -q train_label.zip)   # train_label_blend_seq_full.json, train_label_pdbbind_seq.json
}

download_test() {
  echo "[3/3] Test benchmarks -> test_datasets/"
  mkdir -p test_datasets
  # Small benchmark index files (CASF validation, DUD-E / LIT-PCBA / FEP target lists)
  # from the LigUnity repository.
  for f in casf.lmdb casf_label_seq.json dude.json PCBA.json FEP.json dekois.json \
           fep_assay_ids.json fep_repeat_ligands_can.json fep_similar_ligands_0d5.json; do
    fetch "https://raw.githubusercontent.com/IDEA-XL/LigUnity/main/test_datasets/${f}" "test_datasets/${f}"
  done
  # The FEP-overlap filters are also read from the training directory.
  mkdir -p data
  ln -sf ../test_datasets/fep_assay_ids.json data/fep_assays.json
  ln -sf ../test_datasets/fep_repeat_ligands_can.json data/fep_repeat_ligands_can.json
  ln -sf ../test_datasets/fep_similar_ligands_0d5.json data/fep_similar_ligands_0d5.json

  # DEKOIS 2.0 (LigUnity release, figshare 10.6084/m9.figshare.27967422), used for OOD evaluation
  fetch https://ndownloader.figshare.com/files/50994021 test_datasets/DEKOIS_2.0x.zip
  (cd test_datasets && unzip -n -q DEKOIS_2.0x.zip)   # -> DEKOIS_2.0x/<target>/

  cat <<'EOF'

  DUD-E and LIT-PCBA are distributed by the LigUnity authors through Google Drive:
    https://drive.google.com/drive/folders/1zW1MGpgunynFxTKXC2Q4RgWxZmg6CInV
  Download dude.zip and pcba.zip into test_datasets/ (e.g. with `gdown --folder`), then run:
    cd test_datasets
    unzip -q pcba.zip                       # -> lit_pcba/<TARGET>/
    unzip -q dude.zip                       # -> data/protein/DUD-E/raw/all/<target>/
    mv data/protein/DUD-E/raw/all DUD-E && rm -r data
EOF
}

case "${part}" in
  pretrain) download_pretrain ;;
  train)    download_train ;;
  test)     download_test ;;
  all)      download_pretrain; download_train; download_test ;;
  *) echo "Usage: bash scripts/download_data.sh [all|pretrain|train|test]"; exit 1 ;;
esac
