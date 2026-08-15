#!/usr/bin/env bash
set -euo pipefail

MODE=${1:?usage: array_huvec_mae.sh smoke|full}
case ${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required} in
  0) MODEL=vit_tiny ;;
  1) MODEL=vit_micro ;;
  *) echo "unexpected array index: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac
exec "${REPO:?REPO is required}/scripts/orcd/run_huvec_mae.sh" "$MODEL" "$MODE"
