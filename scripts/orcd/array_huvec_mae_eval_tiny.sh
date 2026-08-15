#!/usr/bin/env bash
set -euo pipefail

case ${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required} in
  0) RUN_NAME=random_init;   RUN_DIR=; RANDOM_INIT=1 ;;
  1) RUN_NAME=vit_tiny;     RUN_DIR=runs/vit_tiny; RANDOM_INIT=0 ;;
  2) RUN_NAME=tiny_mask50;  RUN_DIR=grid/tiny_mask50; RANDOM_INIT=0 ;;
  3) RUN_NAME=tiny_mask90;  RUN_DIR=grid/tiny_mask90; RANDOM_INIT=0 ;;
  4) RUN_NAME=tiny_source4; RUN_DIR=grid/tiny_source4; RANDOM_INIT=0 ;;
  5) RUN_NAME=tiny_source8; RUN_DIR=grid/tiny_source8; RANDOM_INIT=0 ;;
  6) RUN_NAME=tiny_per_image; RUN_DIR=grid/tiny_per_image; RANDOM_INIT=0 ;;
  7) RUN_NAME=tiny_noaug;   RUN_DIR=grid/tiny_noaug; RANDOM_INIT=0 ;;
  *) echo "unexpected Tiny evaluation index: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

FROZEN=${RESULT_ROOT:?RESULT_ROOT is required}/manifests/frozen
test -f "${DATA_ROOT:?DATA_ROOT is required}/EVAL_DATA_COMPLETE.txt"
module purge
module load miniforge/24.3.0-0
export PYTHONUNBUFFERED=1

CHECKPOINT_ARGS=(--random-init --model vit_tiny --split-id primary_fold0 --seed 0)
if [[ $RANDOM_INIT -eq 0 ]]; then
  test -f "$RESULT_ROOT/$RUN_DIR/RESULT.json"
  test -f "$RESULT_ROOT/$RUN_DIR/best_encoder.pt"
  CHECKPOINT_ARGS=(--checkpoint "$RESULT_ROOT/$RUN_DIR/best_encoder.pt")
fi

exec "${ENV_DIR:?ENV_DIR is required}/bin/python" \
  "${REPO:?REPO is required}/scripts/evaluate_rxrx1_huvec_mae.py" \
  --registry "$FROZEN/study_registry.json" \
  --site-manifest "$FROZEN/data/huvec_sites.parquet" \
  --raw-root "$DATA_ROOT/rxrx1-eval" \
  "${CHECKPOINT_ARGS[@]}" \
  --output-dir "$RESULT_ROOT/evaluation_tiny/$RUN_NAME" \
  --batch-size 512 --workers 8
