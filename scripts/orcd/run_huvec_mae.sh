#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:?usage: run_huvec_mae.sh vit_tiny|vit_micro smoke|full}
MODE=${2:?usage: run_huvec_mae.sh vit_tiny|vit_micro smoke|full}
shift 2
VARIANT_ARGS=("$@")
REPO=${REPO:-$HOME/orcd/scratch/moe-batch-effect/worktrees/huvec-mae}
ENV_DIR=${ENV_DIR:-$HOME/orcd/scratch/moe-batch-effect/envs/huvec-mae-py311}
RESULT_ROOT=${RESULT_ROOT:-$HOME/orcd/pool/moe-batch-effect/huvec_mae_pretrain_20260814}
DATA_ROOT=${DATA_ROOT:-$HOME/orcd/scratch/moe-batch-effect/rxrx1-huvec-primary-fold0}
FROZEN=$RESULT_ROOT/manifests/frozen

if [[ "$MODEL" != vit_tiny && "$MODEL" != vit_micro ]]; then
  echo "unknown model: $MODEL" >&2
  exit 2
fi
if [[ "$MODE" != smoke && "$MODE" != full ]]; then
  echo "unknown mode: $MODE" >&2
  exit 2
fi
test -x "$ENV_DIR/bin/python"
test -f "$ENV_DIR/READY"
test -f "$DATA_ROOT/TRANSFER_COMPLETE.txt"
test -f "$FROZEN/study_registry.json"
test -f "$FROZEN/data/huvec_sites.parquet"

module purge
module load miniforge/24.3.0-0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

if [[ "$MODE" == smoke ]]; then
  if [[ -n ${RUN_NAME:-} ]]; then
    OUTPUT=$RESULT_ROOT/grid_smoke/$RUN_NAME
  else
    OUTPUT=$RESULT_ROOT/smoke/$MODEL
  fi
  EXTRA=(--batch-size 64 --workers 4 --max-epochs 1 --min-epochs 1 \
    --warmup-epochs 0 --patience 1 --max-train-steps 4 --encoder-checkpoint-every 1)
else
  if [[ -n ${RUN_NAME:-} ]]; then
    OUTPUT=$RESULT_ROOT/grid/$RUN_NAME
  else
    OUTPUT=$RESULT_ROOT/runs/$MODEL
  fi
  EXTRA=(--batch-size 256 --workers 8 --max-epochs 200 --min-epochs 30 \
    --warmup-epochs 10 --patience 15 --min-delta 0.001 --encoder-checkpoint-every 10)
fi

set +e
"$ENV_DIR/bin/python" "$REPO/scripts/pretrain_rxrx1_huvec_mae.py" \
  --registry "$FROZEN/study_registry.json" \
  --site-manifest "$FROZEN/data/huvec_sites.parquet" \
  --raw-root "$DATA_ROOT/rxrx1" \
  --output-dir "$OUTPUT" \
  --split-id primary_fold0 \
  --model "$MODEL" \
  --mask-ratio 0.75 \
  "${EXTRA[@]}" \
  "${VARIANT_ARGS[@]}"
RC=$?
set -e

if [[ $RC -eq 99 && -n ${SLURM_JOB_ID:-} ]]; then
  echo "Requeueing $SLURM_JOB_ID after a checkpointed scheduler signal"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$RC"
