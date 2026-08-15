#!/usr/bin/env bash
set -euo pipefail

case ${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required} in
  0) RUN_NAME=random_global_anchor;                 RECIPE=huvec_finetune_standard.json; NORM=frozen_global ;;
  1) RUN_NAME=random_per_image_standard;            RECIPE=huvec_finetune_standard.json; NORM=per_image ;;
  2) RUN_NAME=mae_canonical_global_anchor;           RECIPE=huvec_finetune_standard.json; NORM=frozen_global; INIT_RUN=runs/vit_tiny ;;
  3) RUN_NAME=mae_canonical_per_image_standard;      RECIPE=huvec_finetune_standard.json; NORM=per_image;     INIT_RUN=runs/vit_tiny ;;
  4) RUN_NAME=mae_per_image_global_anchor;           RECIPE=huvec_finetune_standard.json; NORM=frozen_global; INIT_RUN=grid/tiny_per_image ;;
  5) RUN_NAME=mae_per_image_matched_standard;        RECIPE=huvec_finetune_standard.json; NORM=per_image;     INIT_RUN=grid/tiny_per_image ;;
  6) RUN_NAME=mae_per_image_matched_lr250e6;         RECIPE=huvec_finetune_lr250e6.json;  NORM=per_image;     INIT_RUN=grid/tiny_per_image ;;
  7) RUN_NAME=mae_canonical_per_image_lr250e6;        RECIPE=huvec_finetune_lr250e6.json;  NORM=per_image;     INIT_RUN=runs/vit_tiny ;;
  *) echo "unexpected matched fine-tuning index: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

FROZEN=${RESULT_ROOT:?RESULT_ROOT is required}/manifests/frozen
test -f "${DATA_ROOT:?DATA_ROOT is required}/HUVEC_DATA_COMPLETE.json"
module load pytorch/2.11.0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

INIT_ARGS=()
if [[ -n ${INIT_RUN:-} ]]; then
  test -f "$RESULT_ROOT/$INIT_RUN/RESULT.json"
  test -f "$RESULT_ROOT/$INIT_RUN/best_encoder.pt"
  INIT_ARGS=(--init-checkpoint "$RESULT_ROOT/$INIT_RUN/best_encoder.pt")
fi

python "${REPO:?REPO is required}/scripts/certify_rxrx1_huvec_recipe.py" \
  --result-root "$RESULT_ROOT/finetune_tiny_perlmutter" \
  --registry "$FROZEN/study_registry.json" \
  --site-manifest "$FROZEN/data/huvec_sites.parquet" \
  --raw-root "$DATA_ROOT/rxrx1" \
  --run-name "$RUN_NAME" --model vit_tiny --split-id primary_fold0 \
  --recipes-json "$REPO/configs/$RECIPE" \
  --normalization-mode "$NORM" \
  --batch-size 128 --num-workers 16 --image-size 224 --seed 0 \
  --eval-every 5 --plateau-patience-evals 4 --plateau-min-delta 0.001 \
  --plateau-min-epochs 30 --plateau-max-epochs 80 \
  "${INIT_ARGS[@]}"
