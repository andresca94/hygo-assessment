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

CONFIG_PATH="${CONFIG_PATH:-ml/training/configs/runpod_4090.yaml}"
RAW_DATA_DIR="${RAW_DATA_DIR:-$ROOT_DIR/data/raw}"
PROCESSED_DATA_DIR="${PROCESSED_DATA_DIR:-$ROOT_DIR/data/processed}"
MASTER_MANIFEST="${MASTER_MANIFEST:-$ROOT_DIR/ml/training/outputs/manifests/master_manifest.csv}"
ENABLE_UTKFACE="${ENABLE_UTKFACE:-1}"
ENABLE_FAIRFACE="${ENABLE_FAIRFACE:-1}"
ENABLE_APPA_REAL="${ENABLE_APPA_REAL:-1}"
ENABLE_NONREAL_EVAL="${ENABLE_NONREAL_EVAL:-1}"

mkdir -p "$RAW_DATA_DIR" "$PROCESSED_DATA_DIR" "$(dirname "$MASTER_MANIFEST")" logs

write_empty_manifest() {
  local output_path="$1"
  python ml/training/scripts/write_empty_manifest.py --output "$output_path"
}

echo "[runpod] Preparing external assets"
bash scripts/runpod_prepare_assets.sh

echo "[runpod] Capturing initial environment report"
bash scripts/runpod_collect_debug_bundle.sh --mode preflight || true

echo "[runpod] Emitting source acquisition instructions"
python ml/training/scripts/download_sources.py \
  --config ml/training/configs/datasets.yaml \
  --raw-root "$RAW_DATA_DIR" \
  --emit-instructions

echo "[runpod] Validating raw dataset folders"
python ml/training/scripts/validate_dataset_sources.py \
  --config ml/training/configs/datasets.yaml \
  --raw-root "$RAW_DATA_DIR"

echo "[runpod] Preparing datasets"
if [[ "$ENABLE_UTKFACE" == "1" ]]; then
  python ml/training/scripts/prepare_utkface.py --raw-dir "$RAW_DATA_DIR/utkface" --output-dir "$PROCESSED_DATA_DIR/utkface"
else
  echo "[runpod] Skipping UTKFace preparation"
  write_empty_manifest "$PROCESSED_DATA_DIR/utkface/manifest.csv"
fi

if [[ "$ENABLE_FAIRFACE" == "1" ]]; then
  python ml/training/scripts/prepare_fairface.py --raw-dir "$RAW_DATA_DIR/fairface" --output-dir "$PROCESSED_DATA_DIR/fairface"
else
  echo "[runpod] Skipping FairFace preparation"
  write_empty_manifest "$PROCESSED_DATA_DIR/fairface/manifest.csv"
fi

if [[ "$ENABLE_APPA_REAL" == "1" ]]; then
  python ml/training/scripts/prepare_appa_real.py --raw-dir "$RAW_DATA_DIR/appa_real" --output-dir "$PROCESSED_DATA_DIR/appa_real"
else
  echo "[runpod] Skipping APPA-REAL preparation"
  write_empty_manifest "$PROCESSED_DATA_DIR/appa_real/manifest.csv"
fi

if [[ "$ENABLE_NONREAL_EVAL" == "1" ]]; then
  python ml/training/scripts/prepare_nonreal_eval.py --raw-dir "$RAW_DATA_DIR/nonreal" --output-dir "$PROCESSED_DATA_DIR/nonreal"
else
  echo "[runpod] Skipping non-real robustness preparation"
  write_empty_manifest "$PROCESSED_DATA_DIR/nonreal/manifest.csv"
fi

echo "[runpod] Building master manifest"
python ml/training/scripts/split_dataset.py \
  --processed-root "$PROCESSED_DATA_DIR" \
  --output-manifest "$MASTER_MANIFEST"

echo "[runpod] Deduplicating manifest"
python ml/training/scripts/deduplicate.py --input-manifest "$MASTER_MANIFEST" --output-manifest "$MASTER_MANIFEST"

echo "[runpod] Validating merged manifest"
python ml/training/scripts/validate_manifest.py --manifest "$MASTER_MANIFEST" --require-val-test

echo "[runpod] Training main model"
python ml/training/scripts/train_main.py --config "$CONFIG_PATH"

echo "[runpod] Training auxiliary model"
python ml/training/scripts/train_aux.py --config "$CONFIG_PATH"

echo "[runpod] Validation evaluation"
python ml/training/scripts/evaluate.py --config "$CONFIG_PATH" --split val

echo "[runpod] Calibration"
python ml/training/scripts/calibrate.py --config "$CONFIG_PATH"

echo "[runpod] Final test evaluation"
python ml/training/scripts/evaluate.py --config "$CONFIG_PATH" --split test

echo "[runpod] Exporting model bundle"
bash scripts/runpod_export_artifacts.sh

echo "[runpod] Collecting final debug bundle"
bash scripts/runpod_collect_debug_bundle.sh --mode final

echo "[runpod] Pipeline completed"
