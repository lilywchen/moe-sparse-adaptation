#!/usr/bin/env bash
set -euo pipefail

case ${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required} in
  0) RUN_NAME=random_standard ;;
  1) RUN_NAME=mae_standard ;;
  2) RUN_NAME=mae_lr250e6 ;;
  3) RUN_NAME=mae_lr100e6 ;;
  *) echo "unexpected fine-tuning evaluation index: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

FROZEN=${RESULT_ROOT:?RESULT_ROOT is required}/manifests/frozen
FINETUNE_ROOT=$RESULT_ROOT/finetune_tiny
FINETUNE_DIR=$FINETUNE_ROOT/recipe_certification/$RUN_NAME/vit_tiny
test -f "$FINETUNE_DIR/PLATEAU_RESULT.json"
module purge
module load miniforge/24.3.0-0
export PYTHONUNBUFFERED=1

exec "${ENV_DIR:?ENV_DIR is required}/bin/python" \
  "${REPO:?REPO is required}/scripts/evaluate_rxrx1_huvec_finetune.py" \
  --finetune-dir "$FINETUNE_DIR" \
  --registry "$FROZEN/study_registry.json" \
  --site-manifest "$FROZEN/data/huvec_sites.parquet" \
  --raw-root "${DATA_ROOT:?DATA_ROOT is required}/rxrx1-eval" \
  --output-dir "$FINETUNE_ROOT/evaluation/$RUN_NAME" \
  --batch-size 512 --workers 8
