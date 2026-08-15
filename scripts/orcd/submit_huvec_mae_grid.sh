#!/usr/bin/env bash
set -euo pipefail

TRANSFER_JOB=${1:?usage: submit_huvec_mae_grid.sh TRANSFER_JOB SETUP_JOB BASELINE_FULL_JOB}
SETUP_JOB=${2:?usage: submit_huvec_mae_grid.sh TRANSFER_JOB SETUP_JOB BASELINE_FULL_JOB}
BASELINE_FULL_JOB=${3:?usage: submit_huvec_mae_grid.sh TRANSFER_JOB SETUP_JOB BASELINE_FULL_JOB}
REPO=${REPO:-$HOME/orcd/scratch/moe-batch-effect/worktrees/huvec-mae}
ENV_DIR=${ENV_DIR:-$HOME/orcd/scratch/moe-batch-effect/envs/huvec-mae-py311}
RESULT_ROOT=${RESULT_ROOT:-$HOME/orcd/pool/moe-batch-effect/huvec_mae_pretrain_20260814}
DATA_ROOT=${DATA_ROOT:-$HOME/orcd/scratch/moe-batch-effect/rxrx1-huvec-primary-fold0}
LOG_ROOT=$RESULT_ROOT/logs
mkdir -p "$LOG_ROOT"
EXPORTS="ALL,REPO=$REPO,ENV_DIR=$ENV_DIR,RESULT_ROOT=$RESULT_ROOT,DATA_ROOT=$DATA_ROOT"

GRID_SMOKE_JOB=$(sbatch --parsable \
  --account=mit_general --partition=mit_normal_gpu --job-name=huvec_grid_smoke \
  --time=01:00:00 --cpus-per-task=8 --mem=64G --gres=gpu:h200:1 \
  --array=0-11%2 --dependency="afterok:$TRANSFER_JOB:$SETUP_JOB" \
  --signal=B:USR1@180 --requeue \
  --output="$LOG_ROOT/grid-smoke-%A_%a.log" --export="$EXPORTS" \
  "$REPO/scripts/orcd/array_huvec_mae_grid.sh" smoke)

GRID_FULL_JOB=$(sbatch --parsable \
  --account=mit_general --partition=mit_normal_gpu --job-name=huvec_grid_full \
  --time=06:00:00 --cpus-per-task=8 --mem=64G --gres=gpu:h200:1 \
  --array=0-11%2 --dependency="afterok:$GRID_SMOKE_JOB" \
  --signal=B:USR1@180 --requeue \
  --output="$LOG_ROOT/grid-full-%A_%a.log" --export="$EXPORTS" \
  "$REPO/scripts/orcd/array_huvec_mae_grid.sh" full)

printf 'BASELINE_FULL_JOB=%s\nGRID_SMOKE_JOB=%s\nGRID_FULL_JOB=%s\n' \
  "$BASELINE_FULL_JOB" "$GRID_SMOKE_JOB" "$GRID_FULL_JOB"
