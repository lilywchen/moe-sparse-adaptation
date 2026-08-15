#!/usr/bin/env bash
set -euo pipefail

BASELINE_FULL_JOB=${1:?usage: submit_huvec_mae_eval.sh BASELINE_FULL GRID_FULL EVAL_DATA}
GRID_FULL_JOB=${2:?usage: submit_huvec_mae_eval.sh BASELINE_FULL GRID_FULL EVAL_DATA}
EVAL_DATA_JOB=${3:?usage: submit_huvec_mae_eval.sh BASELINE_FULL GRID_FULL EVAL_DATA}
REPO=${REPO:-$HOME/orcd/scratch/moe-batch-effect/worktrees/huvec-mae}
ENV_DIR=${ENV_DIR:-$HOME/orcd/scratch/moe-batch-effect/envs/huvec-mae-py311}
RESULT_ROOT=${RESULT_ROOT:-$HOME/orcd/pool/moe-batch-effect/huvec_mae_pretrain_20260814}
DATA_ROOT=${DATA_ROOT:-$HOME/orcd/scratch/moe-batch-effect/rxrx1-huvec-primary-fold0}
LOG_ROOT=$RESULT_ROOT/logs
EXPORTS="ALL,REPO=$REPO,ENV_DIR=$ENV_DIR,RESULT_ROOT=$RESULT_ROOT,DATA_ROOT=$DATA_ROOT"

EVAL_JOB=$(sbatch --parsable \
  --account=mit_general --partition=mit_normal_gpu --job-name=huvec_mae_eval \
  --time=03:00:00 --cpus-per-task=8 --mem=96G --gres=gpu:h200:1 \
  --array=0-13%2 \
  --dependency="afterok:$BASELINE_FULL_JOB:$GRID_FULL_JOB:$EVAL_DATA_JOB" \
  --output="$LOG_ROOT/eval-%A_%a.log" --export="$EXPORTS" \
  "$REPO/scripts/orcd/array_huvec_mae_eval.sh")
AGG_JOB=$(sbatch --parsable \
  --account=mit_general --partition=mit_normal --job-name=huvec_mae_aggregate \
  --time=00:30:00 --cpus-per-task=2 --mem=16G --dependency="afterok:$EVAL_JOB" \
  --output="$LOG_ROOT/aggregate-%j.log" --export="$EXPORTS" \
  --wrap="$ENV_DIR/bin/python $REPO/scripts/aggregate_rxrx1_huvec_mae.py --result-root $RESULT_ROOT")
printf 'EVAL_JOB=%s\nAGG_JOB=%s\n' "$EVAL_JOB" "$AGG_JOB"
