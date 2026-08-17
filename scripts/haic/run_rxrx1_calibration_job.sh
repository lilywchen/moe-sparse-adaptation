#!/usr/bin/env bash
set -euo pipefail
: "${LANE:?}"
: "${FULL:?}"
: "${PYBIN:?}"
: "${EXTRA_SITE:?}"
export PATH="$PYBIN:$PATH"
export PYTHONPATH="$EXTRA_SITE${PYTHONPATH:+:$PYTHONPATH}"

case "$LANE" in
  official)
    export MANIFEST="$FULL/manifests/all_sites.parquet"
    export RUN_NAME=official_densenet161 MODEL=densenet161 NPROC=8
    export PER_GPU_BATCH=16 WORKERS=10 EXTRA_ARGS=--adabn
    ;;
  crossfit)
    FOLD=${SLURM_ARRAY_TASK_ID:?crossfit requires an array task id}
    export MANIFEST="$FULL/huvec_crossfits/huvec_crossfit${FOLD}_h16.parquet"
    export RUN_NAME="huvec_crossfit${FOLD}_h16_densenet" MODEL=densenet161 NPROC=4
    export PER_GPU_BATCH=16 WORKERS=10 SPLIT=custom EXTRA_ARGS=
    ;;
  *) echo "unknown lane: $LANE" >&2; exit 2 ;;
esac
bash "$CODE/scripts/multicluster/run_rxrx1_calibration.sh"

