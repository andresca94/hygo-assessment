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

PYTHON_BIN="${PYTHON_BIN:-python}"
SKIP_TORCH_INSTALL="${SKIP_TORCH_INSTALL:-1}"
FILTERED_DIR="$ROOT_DIR/tmp/filtered_requirements"

prepare_requirements() {
  local source_file="$1"
  local target_file="$2"

  mkdir -p "$FILTERED_DIR"
  if [[ "$SKIP_TORCH_INSTALL" == "1" ]]; then
    grep -vE '^(torch|torchvision)([<>=].*)?$' "$source_file" > "$target_file"
  else
    cp "$source_file" "$target_file"
  fi
}

echo "[deps] Upgrading pip tooling"
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

TRAIN_REQ_FILE="$FILTERED_DIR/training.requirements.txt"
INFER_REQ_FILE="$FILTERED_DIR/inference.requirements.txt"
prepare_requirements "ml/training/requirements.txt" "$TRAIN_REQ_FILE"
prepare_requirements "ml/inference/requirements.txt" "$INFER_REQ_FILE"

if [[ "$SKIP_TORCH_INSTALL" == "1" ]]; then
  echo "[deps] Skipping torch and torchvision installation because the RunPod template already provides PyTorch"
fi

echo "[deps] Installing training dependencies"
"$PYTHON_BIN" -m pip install -r "$TRAIN_REQ_FILE"

echo "[deps] Installing inference dependencies"
"$PYTHON_BIN" -m pip install -r "$INFER_REQ_FILE"

if [[ "${INSTALL_API_DEPS:-0}" == "1" ]]; then
  if command -v npm >/dev/null 2>&1; then
    echo "[deps] Installing NestJS dependencies"
    (cd api/nestjs-service && npm install)
  else
    echo "[deps] npm is not available on this image; skipping NestJS dependency install"
    echo "[deps] You can still train now and install API dependencies later in a Node-capable environment"
  fi
else
  echo "[deps] Skipping npm install because INSTALL_API_DEPS is not enabled"
fi

echo "[deps] Dependency installation completed"
