#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <user@pod-host> [remote_root] [local_target]"
  exit 1
fi

REMOTE_HOST="$1"
REMOTE_ROOT="${2:-/workspace}"
LOCAL_TARGET="${3:-./downloads}"

mkdir -p "$LOCAL_TARGET"

echo "[sync] Pulling debug bundles from $REMOTE_HOST"
scp "$REMOTE_HOST:$REMOTE_ROOT/debug_bundles/*.tar.gz" "$LOCAL_TARGET/" || true

echo "[sync] Pulling exported artifacts from $REMOTE_HOST"
scp "$REMOTE_HOST:$REMOTE_ROOT/exports/*.tar.gz" "$LOCAL_TARGET/" || true

echo "[sync] Done. Files are in $LOCAL_TARGET"
