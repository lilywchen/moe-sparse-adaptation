#!/usr/bin/env bash
set -euo pipefail

LANE=${1:?usage: run_rxrx1_calibration_lane.sh official_densenet|official_resnet|official_vit|crossfit0..5}
CODE=${CODE:-/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation}
PYTHON=${PYTHON:-/home/idies/workspace/Storage/lchen5/persistent/envs/moe/bin/python}
RAW_ROOT=${RAW_ROOT:-/home/idies/workspace/Storage/lchen5/persistent/data/rxrx1_raw/rxrx1}
FROZEN_ROOT=${FROZEN_ROOT:-/home/idies/workspace/Storage/lchen5/persistent/data/rxrx1_calibration_full}
RESULT_ROOT=${RESULT_ROOT:-/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/substrate_rxrx1/rxrx1_calibration_20260817}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the frozen 40-character launch commit}

test -x "$PYTHON"
test -f "$RAW_ROOT/metadata.csv"
test -d "$RAW_ROOT/images"
test "$(git -C "$CODE" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$CODE" status --porcelain --untracked-files=no)" || {
  echo "tracked checkout is dirty: $CODE" >&2
  exit 3
}

mapfile -t GPU_PIDS < <(nvidia-smi --query-compute-apps=pid \
  --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -u)
if (( ${#GPU_PIDS[@]} )); then
  echo "refusing to collide with GPU processes: ${GPU_PIDS[*]}" >&2
  nvidia-smi
  exit 4
fi
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [[ "$GPU_COUNT" -ne 2 ]]; then
  echo "expected a 2-GPU SciServer container, observed $GPU_COUNT" >&2
  exit 4
fi

mkdir -p "$FROZEN_ROOT/manifests" "$FROZEN_ROOT/huvec_crossfits" \
  "$RESULT_ROOT/logs"
exec 9>"$FROZEN_ROOT/.prepare.lock"
flock 9
if [[ ! -s "$FROZEN_ROOT/manifests/all_sites.parquet" ]]; then
  "$PYTHON" "$CODE/scripts/build_rxrx1_full_manifest.py" \
    --metadata-csv "$RAW_ROOT/metadata.csv" --raw-root "$RAW_ROOT" \
    --output-root "$FROZEN_ROOT/manifests"
fi
if [[ ! -s "$FROZEN_ROOT/huvec_crossfits/huvec_crossfit_registry.json" ]]; then
  "$PYTHON" "$CODE/scripts/freeze_rxrx1_huvec_crossfits.py" \
    --all-sites "$FROZEN_ROOT/manifests/all_sites.parquet" \
    --output-root "$FROZEN_ROOT/huvec_crossfits"
fi
flock -u 9

export CODE EXPECTED_COMMIT RAW_ROOT RESULT_ROOT NPROC=2 WORKERS=${WORKERS:-8}
export PATH="$(dirname "$PYTHON"):$PATH"
case "$LANE" in
  official_densenet)
    export MANIFEST="$FROZEN_ROOT/manifests/all_sites.parquet"
    export RUN_NAME=official_densenet161 MODEL=densenet161 PER_GPU_BATCH=16
    export SPLIT=official EXTRA_ARGS=--adabn
    ;;
  official_resnet)
    export MANIFEST="$FROZEN_ROOT/manifests/all_sites.parquet"
    export RUN_NAME=official_resnet50 MODEL=resnet50 PER_GPU_BATCH=32
    export SPLIT=official EXTRA_ARGS=--adabn
    ;;
  official_vit)
    export MANIFEST="$FROZEN_ROOT/manifests/all_sites.parquet"
    export RUN_NAME=official_vit_small MODEL=vit_small PER_GPU_BATCH=8
    export SPLIT=official EXTRA_ARGS=
    ;;
  crossfit[0-5])
    FOLD=${LANE#crossfit}
    export MANIFEST="$FROZEN_ROOT/huvec_crossfits/huvec_crossfit${FOLD}_h16.parquet"
    export RUN_NAME="huvec_crossfit${FOLD}_h16_densenet"
    export MODEL=densenet161 PER_GPU_BATCH=16 SPLIT=custom EXTRA_ARGS=
    ;;
  *)
    echo "unknown lane: $LANE" >&2
    exit 2
    ;;
esac

echo "[sciserver-lane] lane=$LANE run=$RUN_NAME commit=$EXPECTED_COMMIT"
exec bash "$CODE/scripts/multicluster/run_rxrx1_calibration.sh"
