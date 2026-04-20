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
FAIRFACE_DIR="$RAW_DATA_DIR/fairface"
TMP_DIR="${TMP_DIR:-$ROOT_DIR/tmp/dataset_downloads}"

mkdir -p "$FAIRFACE_DIR" "$TMP_DIR"

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

download_gdrive_file() {
  local file_id="$1"
  local output_path="$2"
  python -m gdown --fuzzy "https://drive.google.com/file/d/${file_id}/view?usp=sharing" -O "$output_path"
}

echo "[datasets] Ensuring gdown is available"
ensure_python_package "gdown"

echo "[datasets] Downloading FairFace images from the official dataset links"
download_gdrive_file "1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86" "$TMP_DIR/fairface-img-margin025-trainval.zip"
download_gdrive_file "1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH" "$FAIRFACE_DIR/fairface_label_train.csv"
download_gdrive_file "1wOdja-ezstMEp81tX1a-EYkFebev4h7D" "$FAIRFACE_DIR/fairface_label_val.csv"

echo "[datasets] Extracting FairFace images into $FAIRFACE_DIR"
python_extract_zip "$TMP_DIR/fairface-img-margin025-trainval.zip" "$FAIRFACE_DIR"

echo "[datasets] Validating raw dataset folders"
python ml/training/scripts/validate_dataset_sources.py \
  --config ml/training/configs/datasets.yaml \
  --raw-root "$RAW_DATA_DIR"

echo "[datasets] Minimum dataset staging completed"
echo "[datasets] You can now run: bash scripts/runpod_train_full.sh"
