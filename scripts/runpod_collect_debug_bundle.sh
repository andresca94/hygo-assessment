#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
fi
MODE="${1:-}"
if [[ "$MODE" == "--mode" ]]; then
  MODE="${2:-manual}"
else
  MODE="${MODE:-manual}"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BUNDLE_ROOT="$ROOT_DIR/debug_bundles/${TIMESTAMP}_${MODE}"
mkdir -p "$BUNDLE_ROOT"

copy_if_exists() {
  local source_path="$1"
  local target_path="$2"

  if [[ -e "$source_path" ]]; then
    mkdir -p "$(dirname "$target_path")"
    cp -R "$source_path" "$target_path"
  fi
}

echo "[debug] Writing host and environment information"
uname -a > "$BUNDLE_ROOT/uname.txt" 2>&1 || true
env | sort > "$BUNDLE_ROOT/env.txt" 2>&1 || true
python --version > "$BUNDLE_ROOT/python_version.txt" 2>&1 || true
node --version > "$BUNDLE_ROOT/node_version.txt" 2>&1 || true
npm --version > "$BUNDLE_ROOT/npm_version.txt" 2>&1 || true
nvidia-smi > "$BUNDLE_ROOT/nvidia_smi.txt" 2>&1 || true
df -h > "$BUNDLE_ROOT/disk_usage.txt" 2>&1 || true
free -h > "$BUNDLE_ROOT/memory.txt" 2>&1 || vm_stat > "$BUNDLE_ROOT/memory.txt" 2>&1 || true

echo "[debug] Capturing git context"
git -C "$ROOT_DIR" status --short > "$BUNDLE_ROOT/git_status.txt" 2>&1 || true
git -C "$ROOT_DIR" rev-parse HEAD > "$BUNDLE_ROOT/git_head.txt" 2>&1 || true
git -C "$ROOT_DIR" diff --stat > "$BUNDLE_ROOT/git_diff_stat.txt" 2>&1 || true

echo "[debug] Copying configs and outputs"
copy_if_exists "$ROOT_DIR/.env" "$BUNDLE_ROOT/project/.env"
copy_if_exists "$ROOT_DIR/ml/training/configs" "$BUNDLE_ROOT/project/ml/training/configs"
copy_if_exists "$ROOT_DIR/ml/training/outputs/exported" "$BUNDLE_ROOT/project/ml/training/outputs/exported"
copy_if_exists "$ROOT_DIR/ml/training/outputs/history" "$BUNDLE_ROOT/project/ml/training/outputs/history"
copy_if_exists "$ROOT_DIR/reports" "$BUNDLE_ROOT/project/reports"
copy_if_exists "$ROOT_DIR/logs" "$BUNDLE_ROOT/project/logs"

ARCHIVE_PATH="$ROOT_DIR/debug_bundles/${TIMESTAMP}_${MODE}.tar.gz"
tar -czf "$ARCHIVE_PATH" -C "$ROOT_DIR/debug_bundles" "${TIMESTAMP}_${MODE}"

echo "[debug] Bundle created at $ARCHIVE_PATH"
