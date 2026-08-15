#!/usr/bin/env bash
set -euo pipefail

case ${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required} in
  0) RUN_NAME=random_standard; RECIPE=huvec_finetune_standard.json; USE_MAE=0 ;;
  1) RUN_NAME=mae_standard;    RECIPE=huvec_finetune_standard.json; INIT_RUN=runs/vit_tiny ;;
  2) RUN_NAME=mae_per_image_standard; RECIPE=huvec_finetune_standard.json; INIT_RUN=grid/tiny_per_image ;;
  3) RUN_NAME=mae_per_image_lr250e6; RECIPE=huvec_finetune_lr250e6.json; INIT_RUN=grid/tiny_per_image ;;
  *) echo "unexpected fine-tuning index: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

FROZEN=${RESULT_ROOT:?RESULT_ROOT is required}/manifests/frozen
test -f "${DATA_ROOT:?DATA_ROOT is required}/EVAL_DATA_COMPLETE.txt"
module purge
module load miniforge/24.3.0-0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

INIT_ARGS=()
if [[ -n ${INIT_RUN:-} ]]; then
  test -f "$RESULT_ROOT/$INIT_RUN/RESULT.json"
  test -f "$RESULT_ROOT/$INIT_RUN/best_encoder.pt"
  INIT_ARGS=(--init-checkpoint "$RESULT_ROOT/$INIT_RUN/best_encoder.pt")
fi

exec "${ENV_DIR:?ENV_DIR is required}/bin/python" \
  "${REPO:?REPO is required}/scripts/certify_rxrx1_huvec_recipe.py" \
  --result-root "$RESULT_ROOT/finetune_tiny" \
  --registry "$FROZEN/study_registry.json" \
  --site-manifest "$FROZEN/data/huvec_sites.parquet" \
  --raw-root "$DATA_ROOT/rxrx1-eval" \
  --run-name "$RUN_NAME" --model vit_tiny --split-id primary_fold0 \
  --recipes-json "$REPO/configs/$RECIPE" \
  --batch-size 128 --num-workers 8 --image-size 224 --seed 0 \
  --eval-every 5 --plateau-patience-evals 4 --plateau-min-delta 0.001 \
  --plateau-min-epochs 30 --plateau-max-epochs 80 \
  "${INIT_ARGS[@]}"
