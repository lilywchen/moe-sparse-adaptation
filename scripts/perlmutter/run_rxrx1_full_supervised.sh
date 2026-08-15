#!/bin/bash
set -euo pipefail

ARM=${1:?usage: run_rxrx1_full_supervised.sh random|mae FOLD}
FOLD=${2:-full_fold0}
CODE=${CODE:-/pscratch/sd/l/lilychen/moe-batch-effect/src/moe-sparse-adaptation}
FULL=${FULL:-/pscratch/sd/l/lilychen/moe-batch-effect/data/rxrx1-full}
RESULT=${RESULT:-/pscratch/sd/l/lilychen/moe-batch-effect/results/rxrx1_full_pilot_20260815}

case "$ARM" in
  random) INIT_ARGS=() ;;
  mae) INIT_ARGS=(--init-checkpoint "$RESULT/pretrain/source_only_${FOLD}/vit_tiny/best_encoder.pt") ;;
  *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac

module load pytorch/2.11.0
cd "$CODE"
PYTHONPATH=. python scripts/certify_rxrx1_huvec_recipe.py \
  --result-root "$RESULT" \
  --registry "$FULL/normalization/study_registry.json" \
  --site-manifest "$FULL/manifests/treatment_sites.parquet" \
  --raw-root "$FULL/rxrx1" \
  --run-name "${ARM}_${FOLD}" --model vit_tiny --split-id "$FOLD" \
  --normalization-mode frozen_global "${INIT_ARGS[@]}" \
  --batch-size 128 --num-workers 12 --image-size 224 \
  --eval-every 5 --plateau-patience-evals 4 --plateau-min-delta 0.001 \
  --plateau-min-epochs 30 --plateau-max-epochs 80 --max-epochs 80
