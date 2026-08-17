#!/usr/bin/env bash
set -euo pipefail

CODE=${CODE:-/scratch/users/l1ly/moe-batch-effect/src/moe-sparse-adaptation-calibration}
DATA_ROOT=${DATA_ROOT:-/scratch/users/l1ly/moe-batch-effect/data/rxrx1-full}
ENV=${ENV:-/scratch/users/l1ly/moe-batch-effect/envs/rxrx1-py312}
RESULT_ROOT=${RESULT_ROOT:-/scratch/users/l1ly/moe-batch-effect/results/rxrx1_calibration_20260817}
ACCOUNT=${ACCOUNT:-sanmi}
COMMIT=$(cd "$CODE" && git rev-parse HEAD)

mkdir -p "$DATA_ROOT/logs" "$RESULT_ROOT/logs"
COMMON_EXPORT="CODE=$CODE,DATA_ROOT=$DATA_ROOT,ENV=$ENV,RESULT_ROOT=$RESULT_ROOT,EXPECTED_COMMIT=$COMMIT,EFFECTIVE_BATCH=512,EPOCHS=100"

RUNTIME=$(sbatch --parsable -J rx1_runtime -p normal -A "$ACCOUNT" \
  -c 4 --mem=16G -t 02:00:00 \
  -o "$RESULT_ROOT/logs/runtime-%j.out" -e "$RESULT_ROOT/logs/runtime-%j.err" \
  --export="ALL,CODE=$CODE,ENV=$ENV" \
  "$CODE/scripts/sherlock/setup_rxrx1_runtime.sh")

STAGE=$(sbatch --parsable -J rx1_stage -p normal -A "$ACCOUNT" \
  -c 8 --mem=32G -t 08:00:00 \
  -o "$RESULT_ROOT/logs/stage-%j.out" -e "$RESULT_ROOT/logs/stage-%j.err" \
  --export="ALL,DATA_ROOT=$DATA_ROOT" \
  "$CODE/scripts/sherlock/stage_rxrx1_full.sh")

PREP=$(sbatch --parsable -J rx1_prepare -p normal -A "$ACCOUNT" \
  -c 8 --mem=64G -t 03:00:00 --dependency="afterok:$RUNTIME:$STAGE" \
  -o "$RESULT_ROOT/logs/prepare-%j.out" -e "$RESULT_ROOT/logs/prepare-%j.err" \
  --export="ALL,CODE=$CODE,DATA_ROOT=$DATA_ROOT,ENV=$ENV" \
  "$CODE/scripts/sherlock/prepare_rxrx1_calibration.sh")

GPU_CONSTRAINT='GPU_MEM:32GB|GPU_MEM:48GB|GPU_MEM:80GB'
GPU_COMMON=( -p gpu -A "$ACCOUNT" --qos=normal --constraint="$GPU_CONSTRAINT" \
  --dependency="afterok:$PREP" --requeue --signal=B:USR1@180 )

OFFICIAL_DENSE=$(sbatch --parsable -J rx1_off_dense "${GPU_COMMON[@]}" \
  --gres=gpu:2 -c 16 --mem=128G -t 20:00:00 \
  -o "$RESULT_ROOT/logs/official-dense-%j.out" -e "$RESULT_ROOT/logs/official-dense-%j.err" \
  --export="ALL,$COMMON_EXPORT,LANE=official_densenet" \
  "$CODE/scripts/sherlock/run_rxrx1_calibration_job.sh")

OFFICIAL_RESNET=$(sbatch --parsable -J rx1_off_resnet "${GPU_COMMON[@]}" \
  --gres=gpu:2 -c 16 --mem=128G -t 16:00:00 \
  -o "$RESULT_ROOT/logs/official-resnet-%j.out" -e "$RESULT_ROOT/logs/official-resnet-%j.err" \
  --export="ALL,$COMMON_EXPORT,LANE=official_resnet" \
  "$CODE/scripts/sherlock/run_rxrx1_calibration_job.sh")

OFFICIAL_VIT=$(sbatch --parsable -J rx1_off_vit "${GPU_COMMON[@]}" \
  --gres=gpu:2 -c 16 --mem=128G -t 20:00:00 \
  -o "$RESULT_ROOT/logs/official-vit-%j.out" -e "$RESULT_ROOT/logs/official-vit-%j.err" \
  --export="ALL,$COMMON_EXPORT,LANE=official_vit" \
  "$CODE/scripts/sherlock/run_rxrx1_calibration_job.sh")

CROSSFIT=$(sbatch --parsable -J rx1_huvec_xfit "${GPU_COMMON[@]}" \
  --array=0-5%4 --gres=gpu:2 -c 16 --mem=128G -t 20:00:00 \
  -o "$RESULT_ROOT/logs/crossfit-%A_%a.out" -e "$RESULT_ROOT/logs/crossfit-%A_%a.err" \
  --export="ALL,$COMMON_EXPORT,LANE=crossfit" \
  "$CODE/scripts/sherlock/run_rxrx1_calibration_job.sh")

SCALE=$(sbatch --parsable -J rx1_huvec_scale "${GPU_COMMON[@]}" \
  --array=0-17%2 --gres=gpu:1 -c 8 --mem=64G -t 20:00:00 \
  -o "$RESULT_ROOT/logs/scale-%A_%a.out" -e "$RESULT_ROOT/logs/scale-%A_%a.err" \
  --export="ALL,$COMMON_EXPORT,LANE=scale" \
  "$CODE/scripts/sherlock/run_rxrx1_calibration_job.sh")

printf 'runtime=%s\nstage=%s\nprepare=%s\nofficial_dense=%s\nofficial_resnet=%s\nofficial_vit=%s\ncrossfit=%s\nscale=%s\n' \
  "$RUNTIME" "$STAGE" "$PREP" "$OFFICIAL_DENSE" "$OFFICIAL_RESNET" \
  "$OFFICIAL_VIT" "$CROSSFIT" "$SCALE"
