#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
fi
EXPORT_DIR="${EXPORT_DIR:-$ROOT_DIR/exports/latest}"
MODEL_VERSION="${MODEL_VERSION:-age-safety-v1.0.0}"
EXPORTED_MODEL_DIR="${EXPORTED_MODEL_DIR:-$ROOT_DIR/ml/training/outputs/exported}"

mkdir -p "$EXPORT_DIR"

copy_if_exists() {
  local source_path="$1"
  local target_path="$2"

  if [[ -e "$source_path" ]]; then
    mkdir -p "$(dirname "$target_path")"
    cp -R "$source_path" "$target_path"
  fi
}

copy_if_exists "$EXPORTED_MODEL_DIR" "$EXPORT_DIR/model_bundle"
copy_if_exists "$ROOT_DIR/reports" "$EXPORT_DIR/reports"
copy_if_exists "$ROOT_DIR/DECISIONS.md" "$EXPORT_DIR/DECISIONS.md"
copy_if_exists "$ROOT_DIR/DATA_CARD.md" "$EXPORT_DIR/DATA_CARD.md"

cat > "$EXPORT_DIR/manifest.txt" <<EOF
model_version=$MODEL_VERSION
exported_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
root_dir=$ROOT_DIR
EOF

if [[ -d "$EXPORTED_MODEL_DIR" ]]; then
  cat > "$EXPORT_DIR/model_bundle_manifest.json" <<EOF
{
  "model_version": "$MODEL_VERSION",
  "exported_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "source_dir": "$EXPORTED_MODEL_DIR",
  "required_files": [
    "main_best.pt",
    "aux_best.pt",
    "calibration.json",
    "policy.json"
  ]
}
EOF
  tar -czf "$ROOT_DIR/exports/${MODEL_VERSION}_inference_bundle.tar.gz" -C "$EXPORTED_MODEL_DIR" .
fi

ARCHIVE_PATH="$ROOT_DIR/exports/${MODEL_VERSION}_artifacts.tar.gz"
tar -czf "$ARCHIVE_PATH" -C "$ROOT_DIR/exports/latest" .

echo "[export] Export created at $ARCHIVE_PATH"
