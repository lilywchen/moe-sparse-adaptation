#!/bin/bash
set -euo pipefail

UPSTREAM_JOB=${1:?usage: submit_rxrx1_full_prep.sh SPLIT_JOB_ID}
CODE=${CODE:-/pscratch/sd/l/lilychen/moe-batch-effect/src/moe-sparse-adaptation}
FULL=${FULL:-/pscratch/sd/l/lilychen/moe-batch-effect/data/rxrx1-full}
SHARDS=${SHARDS:-8}

mkdir -p "$FULL/logs"
MOMENT_JOB=$(sbatch --parsable \
  -A m1266 -C cpu -q shared -N1 -n1 -c16 -t 02:00:00 \
  --array="0-$((SHARDS - 1))%${SHARDS}" \
  --job-name=rxrx1_moments --dependency="afterok:${UPSTREAM_JOB}" \
  --output="$FULL/logs/moments-%A_%a.log" \
  --export=ALL,CODE="$CODE",FULL="$FULL",SHARDS="$SHARDS",WORKERS=16 \
  "$CODE/scripts/perlmutter/array_rxrx1_full_moments.sh")

FINAL_JOB=$(sbatch --parsable \
  -A m1266 -C cpu -q shared -N1 -n1 -c8 -t 00:30:00 \
  --job-name=rxrx1_ready --dependency="afterok:${MOMENT_JOB}" \
  --output="$FULL/logs/normalization-finalize-%j.log" \
  --wrap="set -e; module load pytorch/2.11.0; cd $CODE; PYTHONPATH=. python scripts/prepare_rxrx1_full_normalization.py --manifest $FULL/manifests/all_sites.parquet --treatment-manifest $FULL/manifests/treatment_sites.parquet --raw-root $FULL/rxrx1 --split-registry $FULL/splits/split_registry.json --output-root $FULL/normalization --num-shards $SHARDS --finalize")

printf 'MOMENT_JOB=%s\nFINAL_JOB=%s\n' "$MOMENT_JOB" "$FINAL_JOB"
