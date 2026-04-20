#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$ROOT_DIR"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
fi

RAW_DATA_DIR="${RAW_DATA_DIR:-$ROOT_DIR/data/raw}"
NONREAL_DIR="$RAW_DATA_DIR/nonreal"
TMP_DIR="${TMP_DIR:-$ROOT_DIR/tmp/dataset_downloads}"
DOWNLOAD_DEEPFAKEFACE="${DOWNLOAD_DEEPFAKEFACE:-1}"
DOWNLOAD_ICARTOONFACE="${DOWNLOAD_ICARTOONFACE:-1}"

mkdir -p "$NONREAL_DIR" "$TMP_DIR"

ensure_python_package() {
  local package_name="$1"
  if ! python - <<PY >/dev/null 2>&1
import importlib
importlib.import_module("${package_name}")
PY
  then
    python -m pip install "$package_name"
  fi
}

python_extract_zip() {
  local archive_path="$1"
  local output_dir="$2"
  python - "$archive_path" "$output_dir" <<'PY'
import sys
import zipfile
from pathlib import Path

archive = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
output_dir.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(archive) as handle:
    handle.extractall(output_dir)
PY
}

download_gdrive_folder() {
  local folder_id="$1"
  local output_dir="$2"
  python -m gdown --folder "https://drive.google.com/drive/folders/${folder_id}" --continue -O "$output_dir"
}

download_hf_dataset_file() {
  local repo_id="$1"
  local filename="$2"
  local output_path="$3"
  python - "$repo_id" "$filename" "$output_path" <<'PY'
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

repo_id, filename, output_path = sys.argv[1], sys.argv[2], Path(sys.argv[3])
output_path.parent.mkdir(parents=True, exist_ok=True)
downloaded = Path(
    hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        local_dir=str(output_path.parent),
        local_dir_use_symlinks=False,
    )
)
if downloaded.resolve() != output_path.resolve():
    shutil.copy2(downloaded, output_path)
PY
}

echo "[robustness] Ensuring Python download helpers are available"
ensure_python_package "gdown"
ensure_python_package "huggingface_hub"

if [[ "$DOWNLOAD_DEEPFAKEFACE" == "1" ]]; then
  DEEPFAKE_DIR="$NONREAL_DIR/deepfakeface"
  mkdir -p "$DEEPFAKE_DIR"
  echo "[robustness] Downloading DeepFakeFace AI-generated subsets from Hugging Face"
  download_hf_dataset_file "OpenRL/DeepFakeFace" "text2img.zip" "$TMP_DIR/deepfakeface_text2img.zip"
  download_hf_dataset_file "OpenRL/DeepFakeFace" "inpainting.zip" "$TMP_DIR/deepfakeface_inpainting.zip"
  echo "[robustness] Extracting DeepFakeFace subsets"
  python_extract_zip "$TMP_DIR/deepfakeface_text2img.zip" "$DEEPFAKE_DIR/text2img"
  python_extract_zip "$TMP_DIR/deepfakeface_inpainting.zip" "$DEEPFAKE_DIR/inpainting"
fi

if [[ "$DOWNLOAD_ICARTOONFACE" == "1" ]]; then
  ICARTOON_DIR="$NONREAL_DIR/icartoonface"
  mkdir -p "$ICARTOON_DIR"
  echo "[robustness] Downloading iCartoonFace detection dataset from the official Google Drive folder"
  download_gdrive_folder "1ARKrhmGAMwVNr8M9kXgDzMUDhzusLxb7" "$ICARTOON_DIR"
fi

echo "[robustness] Validating raw dataset folders"
python ml/training/scripts/validate_dataset_sources.py \
  --config ml/training/configs/datasets.yaml \
  --raw-root "$RAW_DATA_DIR"

echo "[robustness] Robustness dataset staging completed"
echo "[robustness] You can now run: bash scripts/runpod_run_robustness_eval.sh"
