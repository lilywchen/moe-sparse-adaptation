#!/usr/bin/env bash
set -euo pipefail

MAE_JOB=${1:?usage: submit_huvec_mae_finetune.sh MAE_RESUME_JOB_ID}
REPO=${REPO:-$HOME/orcd/scratch/moe-batch-effect/worktrees/huvec-mae}
ENV_DIR=${ENV_DIR:-$HOME/orcd/scratch/moe-batch-effect/envs/huvec-mae-py311}
RESULT_ROOT=${RESULT_ROOT:-$HOME/orcd/pool/moe-batch-effect/huvec_mae_pretrain_20260814}
DATA_ROOT=${DATA_ROOT:-$HOME/orcd/scratch/moe-batch-effect/rxrx1-huvec-primary-fold0}
LOG_ROOT=$RESULT_ROOT/logs
EXPORTS="ALL,REPO=$REPO,ENV_DIR=$ENV_DIR,RESULT_ROOT=$RESULT_ROOT,DATA_ROOT=$DATA_ROOT"

FINETUNE_JOB=$(sbatch --parsable \
  --account=mit_general --partition=mit_normal_gpu --job-name=huvec_tiny_finetune \
  --time=06:00:00 --cpus-per-task=8 --mem=64G --gres=gpu:h200:1 \
  --array=0-3%4 --dependency="afterok:$MAE_JOB" \
  --output="$LOG_ROOT/tiny-finetune-%A_%a.log" --export="$EXPORTS" \
  "$REPO/scripts/orcd/array_huvec_mae_finetune.sh")

EVAL_JOB=$(sbatch --parsable \
  --account=mit_general --partition=mit_normal_gpu --job-name=huvec_tiny_ft_eval \
  --time=02:00:00 --cpus-per-task=8 --mem=64G --gres=gpu:h200:1 \
  --array=0-3%4 --dependency="afterok:$FINETUNE_JOB" \
  --output="$LOG_ROOT/tiny-ft-eval-%A_%a.log" --export="$EXPORTS" \
  "$REPO/scripts/orcd/array_huvec_mae_finetune_eval.sh")

AGG_JOB=$(sbatch --parsable \
  --account=mit_general --partition=mit_normal --job-name=huvec_tiny_ft_aggregate \
  --time=00:30:00 --cpus-per-task=2 --mem=16G --dependency="afterok:$EVAL_JOB" \
  --output="$LOG_ROOT/tiny-ft-aggregate-%j.log" --export="$EXPORTS" \
  --wrap="$ENV_DIR/bin/python $REPO/scripts/aggregate_rxrx1_huvec_finetune.py --result-root $RESULT_ROOT/finetune_tiny")

printf 'FINETUNE_JOB=%s\nEVAL_JOB=%s\nAGG_JOB=%s\n' "$FINETUNE_JOB" "$EVAL_JOB" "$AGG_JOB"
