#!/usr/bin/env bash
set -euo pipefail

MODE=${1:?usage: array_huvec_mae_grid.sh smoke|full}
case ${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required} in
  0)  RUN_NAME=tiny_mask50;    MODEL=vit_tiny;  ARGS=(--mask-ratio 0.50) ;;
  1)  RUN_NAME=tiny_mask90;    MODEL=vit_tiny;  ARGS=(--mask-ratio 0.90) ;;
  2)  RUN_NAME=micro_mask50;   MODEL=vit_micro; ARGS=(--mask-ratio 0.50) ;;
  3)  RUN_NAME=micro_mask90;   MODEL=vit_micro; ARGS=(--mask-ratio 0.90) ;;
  4)  RUN_NAME=tiny_source4;   MODEL=vit_tiny;  ARGS=(--source-experiment-count 4) ;;
  5)  RUN_NAME=tiny_source8;   MODEL=vit_tiny;  ARGS=(--source-experiment-count 8) ;;
  6)  RUN_NAME=micro_source4;  MODEL=vit_micro; ARGS=(--source-experiment-count 4) ;;
  7)  RUN_NAME=micro_source8;  MODEL=vit_micro; ARGS=(--source-experiment-count 8) ;;
  8)  RUN_NAME=tiny_per_image; MODEL=vit_tiny;  ARGS=(--normalization-mode per_image) ;;
  9)  RUN_NAME=micro_per_image; MODEL=vit_micro; ARGS=(--normalization-mode per_image) ;;
  10) RUN_NAME=tiny_noaug;     MODEL=vit_tiny;  ARGS=(--no-train-augmentation) ;;
  11) RUN_NAME=micro_noaug;    MODEL=vit_micro; ARGS=(--no-train-augmentation) ;;
  *) echo "unexpected grid index: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac
export RUN_NAME
exec "${REPO:?REPO is required}/scripts/orcd/run_huvec_mae.sh" \
  "$MODEL" "$MODE" "${ARGS[@]}"
