#!/usr/bin/env bash
set -euo pipefail

: "${CODE:?set CODE}"
: "${DATA_ROOT:?set DATA_ROOT}"
: "${ENV:?set ENV}"
: "${LANE:?set LANE}"

module load py-pytorch/2.4.1_py312
export PYTHON="$ENV/bin/python"
test -x "$PYTHON"

case "$LANE" in
  official_densenet)
    export MANIFEST="$DATA_ROOT/manifests/all_sites.parquet"
    export RUN_NAME=official_densenet161 MODEL=densenet161 NPROC=2
    export PER_GPU_BATCH=8 WORKERS=8 SPLIT=official EXTRA_ARGS=--adabn
    ;;
  official_resnet)
    export MANIFEST="$DATA_ROOT/manifests/all_sites.parquet"
    export RUN_NAME=official_resnet50 MODEL=resnet50 NPROC=2
    export PER_GPU_BATCH=12 WORKERS=8 SPLIT=official EXTRA_ARGS=--adabn
    ;;
  official_vit)
    export MANIFEST="$DATA_ROOT/manifests/all_sites.parquet"
    export RUN_NAME=official_vit_small MODEL=vit_small NPROC=2
    export PER_GPU_BATCH=8 WORKERS=8 SPLIT=official EXTRA_ARGS=
    ;;
  crossfit)
    FOLD=${SLURM_ARRAY_TASK_ID:?crossfit requires an array task id}
    export MANIFEST="$DATA_ROOT/huvec_crossfits/huvec_crossfit${FOLD}_h16.parquet"
    export RUN_NAME="huvec_crossfit${FOLD}_h16_densenet" MODEL=densenet161 NPROC=2
    export PER_GPU_BATCH=8 WORKERS=8 SPLIT=custom EXTRA_ARGS=
    ;;
  scale)
    FOLD=$((SLURM_ARRAY_TASK_ID / 3))
    POS=$((SLURM_ARRAY_TASK_ID % 3))
    SCALES=(4 8 12)
    H=${SCALES[$POS]}
    export MANIFEST="$DATA_ROOT/huvec_crossfits/huvec_crossfit${FOLD}_h${H}.parquet"
    export RUN_NAME="huvec_crossfit${FOLD}_h${H}_resnet" MODEL=resnet50 NPROC=1
    export PER_GPU_BATCH=16 WORKERS=8 SPLIT=custom EXTRA_ARGS=
    ;;
  *) echo "unknown lane: $LANE" >&2; exit 2 ;;
esac

export RAW_ROOT="$DATA_ROOT/rxrx1"
bash "$CODE/scripts/multicluster/run_rxrx1_calibration.sh"

