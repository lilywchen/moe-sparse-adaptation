#!/usr/bin/env bash
set -euo pipefail

CODE=${CODE:-/pscratch/sd/l/lilychen/moe-batch-effect/src/moe-sparse-adaptation}
FULL=${FULL:-/pscratch/sd/l/lilychen/moe-batch-effect/data/rxrx1-full}
RESULT=${RESULT:-/pscratch/sd/l/lilychen/moe-batch-effect/results/rxrx1_calibration_20260817}
COMMIT=$(git -C "$CODE" rev-parse HEAD)
mkdir -p "$RESULT/logs"

submit() {
  local name=$1 model=$2 batch=$3 hours=$4 extra=$5 dependency=${6:-}
  local dep=()
  if [[ -n "$dependency" ]]; then dep=(--dependency="afterok:$dependency"); fi
  sbatch --parsable -J "$name" -A m1266_g -C gpu -q preempt \
    -N 1 --gpus-per-node=4 --cpus-per-task=64 --time="$hours:00:00" \
    --signal=B:USR1@180 -o "$RESULT/logs/$name-%j.out" \
    -e "$RESULT/logs/$name-%j.err" "${dep[@]}" \
    --wrap="module load pytorch/2.11.0; CODE=$CODE EXPECTED_COMMIT=$COMMIT MANIFEST=$FULL/manifests/all_sites.parquet RAW_ROOT=$FULL/rxrx1 RESULT_ROOT=$RESULT RUN_NAME=$name MODEL=$model NPROC=4 PER_GPU_BATCH=$batch WORKERS=16 EXTRA_ARGS='$extra' bash $CODE/scripts/multicluster/run_rxrx1_calibration.sh"
}

SMOKE=$(submit calibration_smoke densenet161 8 1 "--smoke-steps 2")
DENSE=$(submit official_densenet161 densenet161 16 12 "--adabn" "$SMOKE")
RESNET=$(submit official_resnet50 resnet50 32 10 "--adabn" "$SMOKE")
VIT=$(submit official_vit_small vit_small 8 12 "" "$SMOKE")
printf 'smoke=%s\ndensenet=%s\nresnet=%s\nvit=%s\n' "$SMOKE" "$DENSE" "$RESNET" "$VIT"
