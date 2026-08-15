#!/usr/bin/env bash
set -euo pipefail

MODE=${1:?usage: launch_huvec_finetune_group.sh train|eval}
GROUP=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
case $MODE in
  train) WORKER=${REPO:?REPO is required}/scripts/perlmutter/array_huvec_finetune_matched.sh ;;
  eval)  WORKER=${REPO:?REPO is required}/scripts/perlmutter/array_huvec_finetune_eval.sh ;;
  *) echo "mode must be train|eval" >&2; exit 2 ;;
esac

start=$((GROUP * 4))
pids=()
for offset in 0 1 2 3; do
  task=$((start + offset))
  srun --exclusive --ntasks=1 --cpus-per-task=32 --gpus-per-task=1 \
    --gpu-bind=single:1 \
    env SLURM_ARRAY_TASK_ID="$task" bash "$WORKER" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
