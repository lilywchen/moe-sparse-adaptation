#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/pscratch/sd/l/lilychen/moe-batch-effect/src/moe-sparse-adaptation}
RESULT_ROOT=${RESULT_ROOT:-/pscratch/sd/l/lilychen/moe-batch-effect/results/huvec_mae_pretrain_20260815}
DATA_ROOT=${DATA_ROOT:-/pscratch/sd/l/lilychen/moe-batch-effect/data/rxrx1-huvec-primary-fold0}
LOG_ROOT=$RESULT_ROOT/logs
RUNS=(
  random_global_anchor random_per_image_standard
  mae_canonical_global_anchor mae_canonical_per_image_standard
  mae_per_image_global_anchor mae_per_image_matched_standard
  mae_per_image_matched_lr250e6 mae_canonical_per_image_lr250e6
)

test -f "$DATA_ROOT/HUVEC_DATA_COMPLETE.json"
test -f "$RESULT_ROOT/manifests/frozen/study_registry.json"
test -f "$RESULT_ROOT/runs/vit_tiny/best_encoder.pt"
test -f "$RESULT_ROOT/grid/tiny_per_image/best_encoder.pt"
mkdir -p "$LOG_ROOT"
EXPORTS="ALL,REPO=$REPO,RESULT_ROOT=$RESULT_ROOT,DATA_ROOT=$DATA_ROOT"

FINETUNE_JOB=$(sbatch --parsable \
  --account=m1266_g --constraint=gpu --qos=preempt --nodes=1 --gpus=4 \
  --ntasks=4 --cpus-per-task=32 \
  --job-name=huvec_ft_matched --time=08:00:00 \
  --array=0-1%2 --requeue \
  --output="$LOG_ROOT/perlmutter-ft-%A_%a.log" --export="$EXPORTS" \
  "$REPO/scripts/perlmutter/launch_huvec_finetune_group.sh" train)

EVAL_JOB=$(sbatch --parsable \
  --account=m1266_g --constraint=gpu --qos=preempt --nodes=1 --gpus=4 \
  --ntasks=4 --cpus-per-task=32 \
  --job-name=huvec_ft_eval --time=02:00:00 \
  --array=0-1%2 --requeue --dependency="afterok:$FINETUNE_JOB" \
  --output="$LOG_ROOT/perlmutter-ft-eval-%A_%a.log" --export="$EXPORTS" \
  "$REPO/scripts/perlmutter/launch_huvec_finetune_group.sh" eval)

runs_string=${RUNS[*]}
AGG_JOB=$(sbatch --parsable \
  --account=m1266 --constraint=cpu --qos=regular --nodes=1 --ntasks=1 \
  --job-name=huvec_ft_aggregate --time=00:20:00 --cpus-per-task=4 \
  --dependency="afterok:$EVAL_JOB" \
  --output="$LOG_ROOT/perlmutter-ft-aggregate-%j.log" --export="$EXPORTS" \
  --wrap="module load pytorch/2.11.0; python $REPO/scripts/aggregate_rxrx1_huvec_finetune.py --result-root $RESULT_ROOT/finetune_tiny_perlmutter --runs $runs_string")

printf 'FINETUNE_JOB=%s\nEVAL_JOB=%s\nAGG_JOB=%s\n' \
  "$FINETUNE_JOB" "$EVAL_JOB" "$AGG_JOB"
