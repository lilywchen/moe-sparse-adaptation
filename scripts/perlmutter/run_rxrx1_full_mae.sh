#!/bin/bash
set -euo pipefail

FOLD=${1:-full_fold0}
CODE=${CODE:-/pscratch/sd/l/lilychen/moe-batch-effect/src/moe-sparse-adaptation}
FULL=${FULL:-/pscratch/sd/l/lilychen/moe-batch-effect/data/rxrx1-full}
RESULT=${RESULT:-/pscratch/sd/l/lilychen/moe-batch-effect/results/rxrx1_full_pilot_20260815}

case "$FOLD" in
  full_fold0|full_fold1) SOURCE_EXPERIMENTS=33 ;;
  full_fold2) SOURCE_EXPERIMENTS=36 ;;
  *) echo "unknown frozen full-RxRx1 fold: $FOLD" >&2; exit 2 ;;
esac

module load pytorch/2.11.0
cd "$CODE"
PYTHONPATH=. python scripts/pretrain_rxrx1_huvec_mae.py \
  --registry "$FULL/normalization/study_registry.json" \
  --site-manifest "$FULL/manifests/all_sites.parquet" \
  --raw-root "$FULL/rxrx1" \
  --output-dir "$RESULT/pretrain/source_only_${FOLD}/vit_tiny" \
  --split-id "$FOLD" --source-experiment-count "$SOURCE_EXPERIMENTS" \
  --model vit_tiny --image-size 224 --batch-size 128 --workers 12 \
  --normalization-mode frozen_global --base-lr 1.5e-4 --weight-decay 0.05 \
  --max-epochs 60 --min-epochs 20 --warmup-epochs 5 --patience 8 \
  --min-delta 0.001 --encoder-checkpoint-every 10 --resume
