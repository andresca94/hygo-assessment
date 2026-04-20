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

mkdir -p "$RAW_DATA_DIR" "$PROCESSED_DATA_DIR" "$(dirname "$MASTER_MANIFEST")"

echo "[robustness] Preparing non-real datasets"
python ml/training/scripts/prepare_nonreal_eval.py \
  --raw-dir "$RAW_DATA_DIR/nonreal" \
  --output-dir "$PROCESSED_DATA_DIR/nonreal"

echo "[robustness] Rebuilding master manifest with robustness rows"
python ml/training/scripts/split_dataset.py \
  --processed-root "$PROCESSED_DATA_DIR" \
  --output-manifest "$MASTER_MANIFEST"

python ml/training/scripts/deduplicate.py \
  --input-manifest "$MASTER_MANIFEST" \
  --output-manifest "$MASTER_MANIFEST"

python ml/training/scripts/validate_manifest.py --manifest "$MASTER_MANIFEST" --require-val-test

echo "[robustness] Evaluating robustness split with existing checkpoints"
python ml/training/scripts/evaluate.py --config "$CONFIG_PATH" --split robustness

echo "[robustness] Refreshing exported artifacts"
bash scripts/runpod_export_artifacts.sh

echo "[robustness] Collecting debug bundle"
bash scripts/runpod_collect_debug_bundle.sh --mode robustness

echo "[robustness] Robustness evaluation completed"
