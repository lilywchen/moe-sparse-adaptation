#!/usr/bin/env bash
set -euo pipefail

: "${CODE:?set CODE}"
: "${MANIFEST:?set MANIFEST}"
: "${RAW_ROOT:?set RAW_ROOT}"
: "${RESULT_ROOT:?set RESULT_ROOT}"
: "${RUN_NAME:?set RUN_NAME}"
: "${MODEL:?set MODEL}"
: "${NPROC:?set NPROC to the GPUs used by this process}"

PYTHON=${PYTHON:-$(command -v python)}
test -x "$PYTHON"

: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the frozen launch commit}"
IMAGE_SIZE=${IMAGE_SIZE:-512}
PER_GPU_BATCH=${PER_GPU_BATCH:-16}
WORKERS=${WORKERS:-8}
EPOCHS=${EPOCHS:-100}
EFFECTIVE_BATCH=${EFFECTIVE_BATCH:-512}
SPLIT=${SPLIT:-official}

cd "$CODE"
ACTUAL_COMMIT=$(git rev-parse HEAD)
test "$ACTUAL_COMMIT" = "$EXPECTED_COMMIT" || {
  echo "commit mismatch: expected $EXPECTED_COMMIT, got $ACTUAL_COMMIT" >&2
  exit 3
}
test -f "$MANIFEST"
test -d "$RAW_ROOT/images"
mkdir -p "$RESULT_ROOT/logs"

echo "[start] host=$(hostname) run=$RUN_NAME model=$MODEL world=$NPROC commit=$ACTUAL_COMMIT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# EXTRA_ARGS is controlled by the checked-in scheduler launchers below.
# shellcheck disable=SC2086
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
  scripts/train_rxrx1_calibration.py \
  --manifest "$MANIFEST" --raw-root "$RAW_ROOT" \
  --result-root "$RESULT_ROOT" --run-name "$RUN_NAME" --split "$SPLIT" \
  --model "$MODEL" --image-size "$IMAGE_SIZE" --epochs "$EPOCHS" \
  --per-gpu-batch "$PER_GPU_BATCH" --effective-batch "$EFFECTIVE_BATCH" \
  --workers "$WORKERS" --memory-efficient ${EXTRA_ARGS:-}
