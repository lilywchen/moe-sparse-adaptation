#!/usr/bin/env bash
set -euo pipefail

case ${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required} in
  0) RUN_NAME=random_global_anchor ;;
  1) RUN_NAME=random_per_image_standard ;;
  2) RUN_NAME=mae_canonical_global_anchor ;;
  3) RUN_NAME=mae_canonical_per_image_standard ;;
  4) RUN_NAME=mae_per_image_global_anchor ;;
  5) RUN_NAME=mae_per_image_matched_standard ;;
  6) RUN_NAME=mae_per_image_matched_lr250e6 ;;
  7) RUN_NAME=mae_canonical_per_image_lr250e6 ;;
  *) echo "unexpected matched evaluation index: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

FROZEN=${RESULT_ROOT:?RESULT_ROOT is required}/manifests/frozen
FINETUNE_ROOT=$RESULT_ROOT/finetune_tiny_perlmutter
FINETUNE_DIR=$FINETUNE_ROOT/recipe_certification/$RUN_NAME/vit_tiny
test -f "${DATA_ROOT:?DATA_ROOT is required}/HUVEC_DATA_COMPLETE.json"
test -f "$FINETUNE_DIR/PLATEAU_RESULT.json"
module load pytorch/2.11.0
export PYTHONUNBUFFERED=1

python "${REPO:?REPO is required}/scripts/evaluate_rxrx1_huvec_finetune.py" \
  --finetune-dir "$FINETUNE_DIR" \
  --registry "$FROZEN/study_registry.json" \
  --site-manifest "$FROZEN/data/huvec_sites.parquet" \
  --raw-root "$DATA_ROOT/rxrx1" \
  --output-dir "$FINETUNE_ROOT/evaluation/$RUN_NAME" \
  --batch-size 512 --workers 16
