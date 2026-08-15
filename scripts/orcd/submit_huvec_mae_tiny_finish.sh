#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-$HOME/orcd/scratch/moe-batch-effect/worktrees/huvec-mae}
ENV_DIR=${ENV_DIR:-$HOME/orcd/scratch/moe-batch-effect/envs/huvec-mae-py311}
RESULT_ROOT=${RESULT_ROOT:-$HOME/orcd/pool/moe-batch-effect/huvec_mae_pretrain_20260814}
DATA_ROOT=${DATA_ROOT:-$HOME/orcd/scratch/moe-batch-effect/rxrx1-huvec-primary-fold0}
LOG_ROOT=$RESULT_ROOT/logs
mkdir -p "$LOG_ROOT"
EXPORTS="ALL,REPO=$REPO,ENV_DIR=$ENV_DIR,RESULT_ROOT=$RESULT_ROOT,DATA_ROOT=$DATA_ROOT"

RESUME_JOB=$(sbatch --parsable \
  --account=mit_general --partition=mit_normal_gpu --job-name=huvec_tiny_resume \
  --time=06:00:00 --cpus-per-task=8 --mem=64G --gres=gpu:h200:1 \
  --signal=B:USR1@180 --requeue \
  --output="$LOG_ROOT/tiny-resume-%j.log" --export="$EXPORTS" \
  "$REPO/scripts/orcd/run_huvec_mae.sh" vit_tiny full)

EVAL_JOB=$(sbatch --parsable \
  --account=mit_general --partition=mit_normal_gpu --job-name=huvec_tiny_eval \
  --time=03:00:00 --cpus-per-task=8 --mem=96G --gres=gpu:h200:1 \
  --array=0-7%8 --dependency="afterok:$RESUME_JOB" \
  --output="$LOG_ROOT/tiny-eval-%A_%a.log" --export="$EXPORTS" \
  "$REPO/scripts/orcd/array_huvec_mae_eval_tiny.sh")

AGG_JOB=$(sbatch --parsable \
  --account=mit_general --partition=mit_normal --job-name=huvec_tiny_aggregate \
  --time=00:30:00 --cpus-per-task=2 --mem=16G --dependency="afterok:$EVAL_JOB" \
  --output="$LOG_ROOT/tiny-aggregate-%j.log" --export="$EXPORTS" \
  --wrap="$ENV_DIR/bin/python $REPO/scripts/aggregate_rxrx1_huvec_mae_tiny.py --result-root $RESULT_ROOT")

printf 'RESUME_JOB=%s\nEVAL_JOB=%s\nAGG_JOB=%s\n' "$RESUME_JOB" "$EVAL_JOB" "$AGG_JOB"
