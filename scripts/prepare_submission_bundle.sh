#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
fi
MODEL_VERSION="${MODEL_VERSION:-age-safety-v1.0.0}"
SUBMISSION_ROOT="${SUBMISSION_ROOT:-$ROOT_DIR/exports/submission}"
INCLUDE_WEIGHTS="${INCLUDE_WEIGHTS:-0}"
EXPORTED_MODEL_DIR="${EXPORTED_MODEL_DIR:-$ROOT_DIR/ml/training/outputs/exported}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
TARGET_DIR="$SUBMISSION_ROOT/${TIMESTAMP}_${MODEL_VERSION}"
mkdir -p "$TARGET_DIR"

copy_if_exists() {
  local source_path="$1"
  local target_path="$2"
  if [[ -e "$source_path" ]]; then
    mkdir -p "$(dirname "$target_path")"
    cp -R "$source_path" "$target_path"
  fi
}

copy_if_exists "$ROOT_DIR/README.md" "$TARGET_DIR/README.md"
copy_if_exists "$ROOT_DIR/DECISIONS.md" "$TARGET_DIR/DECISIONS.md"
copy_if_exists "$ROOT_DIR/DATA_CARD.md" "$TARGET_DIR/DATA_CARD.md"
copy_if_exists "$ROOT_DIR/RUNPOD_CHECKLIST.md" "$TARGET_DIR/RUNPOD_CHECKLIST.md"
copy_if_exists "$ROOT_DIR/.env.example" "$TARGET_DIR/.env.example"
copy_if_exists "$ROOT_DIR/docker-compose.yml" "$TARGET_DIR/docker-compose.yml"
copy_if_exists "$ROOT_DIR/api" "$TARGET_DIR/api"
copy_if_exists "$ROOT_DIR/ml" "$TARGET_DIR/ml"
copy_if_exists "$ROOT_DIR/scripts" "$TARGET_DIR/scripts"

if [[ "$INCLUDE_WEIGHTS" == "1" && -d "$EXPORTED_MODEL_DIR" ]]; then
  copy_if_exists "$EXPORTED_MODEL_DIR" "$TARGET_DIR/ml/training/outputs/exported"
fi

cat > "$TARGET_DIR/SUBMISSION_NOTES.md" <<EOF
# Submission Notes

Model version: $MODEL_VERSION

Reviewer options:

1. If this bundle includes \`ml/training/outputs/exported\`, inference can run directly after dependency install.
2. If weights are not included, run:

\`\`\`bash
bash scripts/install_inference_bundle.sh <bundle.tar.gz-or-url>
\`\`\`

Then start the services with the normal README instructions.
EOF

ARCHIVE_PATH="$SUBMISSION_ROOT/${TIMESTAMP}_${MODEL_VERSION}_submission.tar.gz"
tar -czf "$ARCHIVE_PATH" -C "$SUBMISSION_ROOT" "$(basename "$TARGET_DIR")"

echo "[submission] Created $ARCHIVE_PATH"
