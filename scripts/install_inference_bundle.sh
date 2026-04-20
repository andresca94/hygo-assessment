#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
fi
TARGET_DIR="${TARGET_DIR:-$ROOT_DIR/ml/training/outputs/exported}"
MODEL_BUNDLE_SOURCE="${1:-${MODEL_BUNDLE_SOURCE:-}}"

if [[ -z "$MODEL_BUNDLE_SOURCE" ]]; then
  echo "Usage: $0 <bundle.tar.gz-or-url>"
  echo "You can also set MODEL_BUNDLE_SOURCE."
  exit 1
fi

mkdir -p "$TARGET_DIR"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

if [[ "$MODEL_BUNDLE_SOURCE" =~ ^https?:// ]]; then
  echo "[install] Downloading inference bundle from URL"
  curl -L "$MODEL_BUNDLE_SOURCE" -o "$TMP_DIR/inference_bundle.tar.gz"
  BUNDLE_PATH="$TMP_DIR/inference_bundle.tar.gz"
else
  BUNDLE_PATH="$MODEL_BUNDLE_SOURCE"
fi

if [[ ! -f "$BUNDLE_PATH" ]]; then
  echo "[install] Bundle not found: $BUNDLE_PATH"
  exit 1
fi

echo "[install] Extracting to $TARGET_DIR"
tar -xzf "$BUNDLE_PATH" -C "$TARGET_DIR"

for required_file in main_best.pt aux_best.pt calibration.json policy.json; do
  if [[ ! -f "$TARGET_DIR/$required_file" ]]; then
    echo "[install] Missing required file after extraction: $required_file"
    exit 1
  fi
done

echo "[install] Inference bundle installed successfully"
