#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$ROOT_DIR"

echo "[runpod] Training recommended final submission candidate"
echo "[runpod] Dataset selection: FairFace supervision + non-real robustness, UTKFace disabled"

ENABLE_UTKFACE=0 \
ENABLE_FAIRFACE=1 \
ENABLE_APPA_REAL=0 \
ENABLE_NONREAL_EVAL=1 \
bash scripts/runpod_train_full.sh
