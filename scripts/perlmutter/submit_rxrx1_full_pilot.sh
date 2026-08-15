#!/bin/bash
set -euo pipefail

READY_JOB=${1:?usage: submit_rxrx1_full_pilot.sh FULL_READY_JOB_ID [FOLD]}
FOLD=${2:-full_fold0}
CODE=${CODE:-/pscratch/sd/l/lilychen/moe-batch-effect/src/moe-sparse-adaptation}
FULL=${FULL:-/pscratch/sd/l/lilychen/moe-batch-effect/data/rxrx1-full}
RESULT=${RESULT:-/pscratch/sd/l/lilychen/moe-batch-effect/results/rxrx1_full_pilot_20260815}
mkdir -p "$RESULT/logs"

submit_gpu() {
  local name=$1 dependency=$2 script=$3
  shift 3
  sbatch --parsable -A m1266_g -C gpu -q preempt -N1 -G1 -c24 -t 08:00:00 \
    --requeue --job-name="$name" \
    --dependency="afterok:${dependency}" --output="$RESULT/logs/${name}-%j.log" \
    --export=ALL,CODE="$CODE",FULL="$FULL",RESULT="$RESULT" "$script" "$@"
}

RANDOM_JOB=$(submit_gpu "full_random_${FOLD}" "$READY_JOB" \
  "$CODE/scripts/perlmutter/run_rxrx1_full_supervised.sh" random "$FOLD")
MAE_JOB=$(submit_gpu "full_mae_${FOLD}" "$READY_JOB" \
  "$CODE/scripts/perlmutter/run_rxrx1_full_mae.sh" "$FOLD")
MAE_FT_JOB=$(submit_gpu "full_maeft_${FOLD}" "$MAE_JOB" \
  "$CODE/scripts/perlmutter/run_rxrx1_full_supervised.sh" mae "$FOLD")
RANDOM_EVAL_JOB=$(submit_gpu "full_reval_${FOLD}" "$RANDOM_JOB" \
  "$CODE/scripts/perlmutter/run_rxrx1_full_eval.sh" random "$FOLD")
MAE_EVAL_JOB=$(submit_gpu "full_meval_${FOLD}" "$MAE_FT_JOB" \
  "$CODE/scripts/perlmutter/run_rxrx1_full_eval.sh" mae "$FOLD")
AGG_JOB=$(sbatch --parsable -A m1266 -C cpu -q shared -N1 -n1 -c4 -t 00:15:00 \
  --job-name="full_agg_${FOLD}" \
  --dependency="afterok:${RANDOM_EVAL_JOB}:${MAE_EVAL_JOB}" \
  --output="$RESULT/logs/full_agg_${FOLD}-%j.log" \
  --wrap="set -e; module load pytorch/2.11.0; cd $CODE; PYTHONPATH=. python scripts/aggregate_rxrx1_full_pilot.py --result-root $RESULT")

printf 'RANDOM_JOB=%s\nMAE_JOB=%s\nMAE_FT_JOB=%s\nRANDOM_EVAL_JOB=%s\nMAE_EVAL_JOB=%s\nAGG_JOB=%s\n' \
  "$RANDOM_JOB" "$MAE_JOB" "$MAE_FT_JOB" "$RANDOM_EVAL_JOB" "$MAE_EVAL_JOB" "$AGG_JOB"
