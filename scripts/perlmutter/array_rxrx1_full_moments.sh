#!/bin/bash
set -euo pipefail

CODE=${CODE:-/pscratch/sd/l/lilychen/moe-batch-effect/src/moe-sparse-adaptation}
FULL=${FULL:-/pscratch/sd/l/lilychen/moe-batch-effect/data/rxrx1-full}
SHARDS=${SHARDS:-8}
WORKERS=${WORKERS:-16}

module load pytorch/2.11.0
cd "$CODE"
PYTHONPATH=. python scripts/prepare_rxrx1_full_normalization.py \
  --manifest "$FULL/manifests/all_sites.parquet" \
  --raw-root "$FULL/rxrx1" \
  --output-root "$FULL/normalization" \
  --num-shards "$SHARDS" \
  --shard-index "${SLURM_ARRAY_TASK_ID:?SLURM array index is required}" \
  --workers "$WORKERS"
