#!/bin/bash
set -euo pipefail

ARM=${1:?usage: run_rxrx1_full_eval.sh random|mae FOLD}
FOLD=${2:-full_fold0}
CODE=${CODE:-/pscratch/sd/l/lilychen/moe-batch-effect/src/moe-sparse-adaptation}
FULL=${FULL:-/pscratch/sd/l/lilychen/moe-batch-effect/data/rxrx1-full}
RESULT=${RESULT:-/pscratch/sd/l/lilychen/moe-batch-effect/results/rxrx1_full_pilot_20260815}

module load pytorch/2.11.0
cd "$CODE"
PYTHONPATH=. python scripts/evaluate_rxrx1_huvec_finetune.py \
  --finetune-dir "$RESULT/recipe_certification/${ARM}_${FOLD}/vit_tiny" \
  --registry "$FULL/normalization/study_registry.json" \
  --site-manifest "$FULL/manifests/treatment_sites.parquet" \
  --raw-root "$FULL/rxrx1" \
  --output-dir "$RESULT/evaluation/${ARM}_${FOLD}" \
  --batch-size 512 --workers 12
