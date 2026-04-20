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

ASSET_CONFIG="${ASSET_CONFIG:-ml/training/configs/model_assets.yaml}"
DOWNLOAD_OPTIONAL_MODEL_ASSETS="${DOWNLOAD_OPTIONAL_MODEL_ASSETS:-0}"

mkdir -p assets third_party .cache/torch .cache/huggingface

echo "[assets] Preparing directories and instruction manifests"
python ml/training/scripts/bootstrap_model_assets.py \
  --config "$ASSET_CONFIG" \
  --emit-instructions \
  --prepare-dirs

if [[ "${AUTO_DOWNLOAD_PUBLIC_ASSETS:-0}" == "1" ]]; then
  echo "[assets] Downloading public assets and cloning external repos"
  EXTRA_ARGS=()
  if [[ "$DOWNLOAD_OPTIONAL_MODEL_ASSETS" == "1" ]]; then
    EXTRA_ARGS+=(--include-optional)
  fi
  python ml/training/scripts/bootstrap_model_assets.py \
    --config "$ASSET_CONFIG" \
    --clone-repos \
    --download-public \
    --prewarm-dinov2 \
    "${EXTRA_ARGS[@]}"
else
  echo "[assets] Skipping downloads because AUTO_DOWNLOAD_PUBLIC_ASSETS is not enabled"
fi

echo "[assets] Asset preparation completed"
