#!/usr/bin/env bash
set -euo pipefail

case ${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required} in
  0)  RUN_NAME=vit_tiny;       RUN_DIR=runs/vit_tiny ;;
  1)  RUN_NAME=vit_micro;      RUN_DIR=runs/vit_micro ;;
  2)  RUN_NAME=tiny_mask50;    RUN_DIR=grid/tiny_mask50 ;;
  3)  RUN_NAME=tiny_mask90;    RUN_DIR=grid/tiny_mask90 ;;
  4)  RUN_NAME=micro_mask50;   RUN_DIR=grid/micro_mask50 ;;
  5)  RUN_NAME=micro_mask90;   RUN_DIR=grid/micro_mask90 ;;
  6)  RUN_NAME=tiny_source4;   RUN_DIR=grid/tiny_source4 ;;
  7)  RUN_NAME=tiny_source8;   RUN_DIR=grid/tiny_source8 ;;
  8)  RUN_NAME=micro_source4;  RUN_DIR=grid/micro_source4 ;;
  9)  RUN_NAME=micro_source8;  RUN_DIR=grid/micro_source8 ;;
  10) RUN_NAME=tiny_per_image; RUN_DIR=grid/tiny_per_image ;;
  11) RUN_NAME=micro_per_image; RUN_DIR=grid/micro_per_image ;;
  12) RUN_NAME=tiny_noaug;     RUN_DIR=grid/tiny_noaug ;;
  13) RUN_NAME=micro_noaug;    RUN_DIR=grid/micro_noaug ;;
  *) echo "unexpected evaluation index: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

FROZEN=${RESULT_ROOT:?RESULT_ROOT is required}/manifests/frozen
test -f "${DATA_ROOT:?DATA_ROOT is required}/EVAL_DATA_COMPLETE.txt"
test -f "$RESULT_ROOT/$RUN_DIR/best_encoder.pt"
module purge
module load miniforge/24.3.0-0
export PYTHONUNBUFFERED=1
exec "${ENV_DIR:?ENV_DIR is required}/bin/python" \
  "${REPO:?REPO is required}/scripts/evaluate_rxrx1_huvec_mae.py" \
  --registry "$FROZEN/study_registry.json" \
  --site-manifest "$FROZEN/data/huvec_sites.parquet" \
  --raw-root "$DATA_ROOT/rxrx1-eval" \
  --checkpoint "$RESULT_ROOT/$RUN_DIR/best_encoder.pt" \
  --output-dir "$RESULT_ROOT/evaluation/$RUN_NAME" \
  --batch-size 512 --workers 8
