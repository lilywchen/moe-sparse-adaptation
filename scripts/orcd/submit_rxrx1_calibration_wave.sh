#!/usr/bin/env bash
set -euo pipefail

STAGE_JOB=${1:?usage: submit_rxrx1_calibration_wave.sh STAGE_JOB_ID}
CODE=${CODE:-/orcd/scratch/orcd/002/l1ly/moe-batch-effect/src/moe-sparse-adaptation}
FULL=${FULL:-/orcd/scratch/orcd/002/l1ly/moe-batch-effect/rxrx1-full}
RESULT=${RESULT:-/orcd/pool/004/l1ly/moe-batch-effect/rxrx1_calibration_20260817}
ENV=${ENV:-/orcd/scratch/orcd/002/l1ly/moe-batch-effect/envs/huvec-mae-py311}
mkdir -p "$RESULT/logs"

COMMON="source $ENV/bin/activate; CODE=$CODE RAW_ROOT=$FULL/rxrx1 RESULT_ROOT=$RESULT"
OFFICIAL=$(sbatch --parsable -J rx1_dense_official -p mit_normal_gpu -A mit_general \
  --gres=gpu:h200:8 --cpus-per-task=64 --mem=0 -t 06:00:00 \
  --dependency="afterok:$STAGE_JOB" --signal=B:USR1@180 \
  -o "$RESULT/logs/official-%j.out" -e "$RESULT/logs/official-%j.err" \
  --wrap="$COMMON MANIFEST=$FULL/manifests/all_sites.parquet RUN_NAME=official_densenet161 MODEL=densenet161 NPROC=8 PER_GPU_BATCH=16 WORKERS=8 EXTRA_ARGS='--adabn' bash $CODE/scripts/multicluster/run_rxrx1_calibration.sh")

CROSSFIT=$(sbatch --parsable -J rx1_huvec_xfit -p mit_preemptable -A mit_general \
  --array=0-5 --gres=gpu:h200:4 --cpus-per-task=48 --mem=0 -t 1-12:00:00 \
  --dependency="afterok:$STAGE_JOB" --requeue --signal=B:USR1@180 \
  -o "$RESULT/logs/crossfit-%A_%a.out" -e "$RESULT/logs/crossfit-%A_%a.err" \
  --wrap="$COMMON MANIFEST=$FULL/huvec_crossfits/huvec_crossfit\${SLURM_ARRAY_TASK_ID}_h16.parquet RUN_NAME=huvec_crossfit\${SLURM_ARRAY_TASK_ID}_h16_densenet MODEL=densenet161 NPROC=4 PER_GPU_BATCH=16 WORKERS=10 SPLIT=custom EXTRA_ARGS='' bash $CODE/scripts/multicluster/run_rxrx1_calibration.sh")

SCALE=$(sbatch --parsable -J rx1_huvec_scale -p mit_preemptable -A mit_general \
  --array=0-17 --gres=gpu:h200:1 --cpus-per-task=12 --mem=96G -t 1-12:00:00 \
  --dependency="afterok:$STAGE_JOB" --requeue --signal=B:USR1@180 \
  -o "$RESULT/logs/scale-%A_%a.out" -e "$RESULT/logs/scale-%A_%a.err" \
  --wrap='set -euo pipefail; FOLD=$((SLURM_ARRAY_TASK_ID/3)); POS=$((SLURM_ARRAY_TASK_ID%3)); SCALES=(4 8 12); H=${SCALES[$POS]}; '"$COMMON"' MANIFEST='"$FULL"'/huvec_crossfits/huvec_crossfit${FOLD}_h${H}.parquet RUN_NAME=huvec_crossfit${FOLD}_h${H}_resnet MODEL=resnet50 NPROC=1 PER_GPU_BATCH=32 WORKERS=10 SPLIT=custom EXTRA_ARGS="" bash '"$CODE"'/scripts/multicluster/run_rxrx1_calibration.sh')

printf 'official=%s\ncrossfit=%s\nscale=%s\n' "$OFFICIAL" "$CROSSFIT" "$SCALE"
