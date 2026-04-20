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
UTKFACE_DIR="$RAW_DATA_DIR/utkface"
TMP_DIR="${TMP_DIR:-$ROOT_DIR/tmp/dataset_downloads}"
UTKFACE_TMP_DIR="$TMP_DIR/utkface_official"
UTKFACE_FOLDER_URL="${UTKFACE_FOLDER_URL:-https://drive.google.com/drive/folders/0BxYys69jI14kU0I1YUQyY1ZDRUE}"

mkdir -p "$UTKFACE_DIR" "$UTKFACE_TMP_DIR"

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

dir_has_images() {
  local search_dir="$1"
  find "$search_dir" -type f \( \
    -iname '*.jpg' -o \
    -iname '*.jpeg' -o \
    -iname '*.png' -o \
    -iname '*.webp' \
  \) -print -quit | grep -q .
}

python_extract_archive() {
  local archive_path="$1"
  local output_dir="$2"
  python - "$archive_path" "$output_dir" <<'PY'
import sys
import tarfile
import zipfile
from pathlib import Path

archive = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
output_dir.mkdir(parents=True, exist_ok=True)
suffixes = "".join(archive.suffixes[-2:]).lower()

if suffixes == ".tar.gz" or archive.suffix.lower() == ".tgz":
    with tarfile.open(archive, mode="r:gz") as handle:
        handle.extractall(output_dir)
elif archive.suffix.lower() == ".zip":
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(output_dir)
else:
    raise RuntimeError(f"Unsupported archive type: {archive}")
PY
}

echo "[datasets] Ensuring gdown is available"
ensure_python_package "gdown"

if dir_has_images "$UTKFACE_DIR"; then
  echo "[datasets] Skipping UTKFace download; images already present in $UTKFACE_DIR"
else
  echo "[datasets] Downloading UTKFace aligned-and-cropped archive from the official dataset folder"
  python -m gdown --folder "$UTKFACE_FOLDER_URL" --continue -O "$UTKFACE_TMP_DIR"

  archive_path=""
  if [[ -f "$UTKFACE_TMP_DIR/UTKFace.tar.gz" ]]; then
    archive_path="$UTKFACE_TMP_DIR/UTKFace.tar.gz"
  else
    archive_path="$(find "$UTKFACE_TMP_DIR" -maxdepth 2 -type f \( -iname 'UTKFace.tar.gz' -o -iname '*.tgz' -o -iname '*.zip' -o -iname '*.tar.gz' \) | head -1)"
  fi

  if [[ -z "$archive_path" ]]; then
    echo "[datasets] Failed to locate a UTKFace archive under $UTKFACE_TMP_DIR" >&2
    exit 1
  fi

  echo "[datasets] Extracting $(basename "$archive_path") into $UTKFACE_DIR"
  python_extract_archive "$archive_path" "$UTKFACE_DIR"
fi

echo "[datasets] Validating raw dataset folders"
python ml/training/scripts/validate_dataset_sources.py \
  --config ml/training/configs/datasets.yaml \
  --raw-root "$RAW_DATA_DIR"

echo "[datasets] UTKFace staging completed"
echo "[datasets] You can now rerun: bash scripts/runpod_train_full.sh"
