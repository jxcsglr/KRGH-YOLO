#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/train_scb.sh DATASET_ROOT [DEVICE] [extra train.py args]"
  exit 2
fi

DATASET_ROOT=$1
DEVICE=${2:-0}
shift
if [[ $# -gt 0 ]]; then shift; fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

python train.py "$DATASET_ROOT" --preset scb --device "$DEVICE" "$@"
