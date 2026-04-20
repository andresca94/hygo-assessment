#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
fi

API_PORT="${API_PORT:-3000}"
ML_PORT="${ML_INFERENCE_PORT:-8000}"

echo "[smoke] Checking inference health"
curl -fsS "http://127.0.0.1:${ML_PORT}/health"
echo

echo "[smoke] Checking public API health"
curl -fsS "http://127.0.0.1:${API_PORT}/v1/age-safety/health"
echo

echo "[smoke] Smoke test completed"
