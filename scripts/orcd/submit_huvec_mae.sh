#!/usr/bin/env bash
set -euo pipefail

TRANSFER_JOB=${1:?usage: submit_huvec_mae.sh TRANSFER_JOB_ID}
REPO=${REPO:-$HOME/orcd/scratch/moe-batch-effect/worktrees/huvec-mae}
ENV_DIR=${ENV_DIR:-$HOME/orcd/scratch/moe-batch-effect/envs/huvec-mae-py311}
RESULT_ROOT=${RESULT_ROOT:-$HOME/orcd/pool/moe-batch-effect/huvec_mae_pretrain_20260814}
DATA_ROOT=${DATA_ROOT:-$HOME/orcd/scratch/moe-batch-effect/rxrx1-huvec-primary-fold0}
LOG_ROOT=$RESULT_ROOT/logs
mkdir -p "$LOG_ROOT"

EXPORTS="ALL,REPO=$REPO,ENV_DIR=$ENV_DIR,RESULT_ROOT=$RESULT_ROOT,DATA_ROOT=$DATA_ROOT"
SETUP_JOB=$(sbatch --parsable \
  --account=mit_general --partition=mit_normal --job-name=huvec_mae_env \
  --time=02:00:00 --cpus-per-task=4 --mem=24G \
  --output="$LOG_ROOT/setup-%j.log" --export="$EXPORTS" \
  "$REPO/scripts/orcd/setup_huvec_mae_env.sh")

SMOKE_JOB=$(sbatch --parsable \
  --account=mit_general --partition=mit_normal_gpu --job-name=huvec_mae_smoke \
  --time=01:00:00 --cpus-per-task=8 --mem=64G --gres=gpu:h100:1 \
  --array=0-1%2 --dependency="afterok:$TRANSFER_JOB:$SETUP_JOB" \
  --signal=B:USR1@180 --requeue \
  --output="$LOG_ROOT/smoke-%A_%a.log" --export="$EXPORTS" \
  "$REPO/scripts/orcd/array_huvec_mae.sh" smoke)

FULL_JOB=$(sbatch --parsable \
  --account=mit_general --partition=mit_normal_gpu --job-name=huvec_mae_full \
  --time=06:00:00 --cpus-per-task=8 --mem=64G --gres=gpu:h100:1 \
  --array=0-1%2 --dependency="afterok:$SMOKE_JOB" \
  --signal=B:USR1@180 --requeue \
  --output="$LOG_ROOT/full-%A_%a.log" --export="$EXPORTS" \
  "$REPO/scripts/orcd/array_huvec_mae.sh" full)

printf 'TRANSFER_JOB=%s\nSETUP_JOB=%s\nSMOKE_JOB=%s\nFULL_JOB=%s\n' \
  "$TRANSFER_JOB" "$SETUP_JOB" "$SMOKE_JOB" "$FULL_JOB"
