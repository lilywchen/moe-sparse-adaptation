#!/usr/bin/env bash
set -euo pipefail

STAGE_JOB=${1:?usage: submit_rxrx1_calibration_wave.sh STAGE_JOB_ID}
BASE=${BASE:-/hai/scratch/l1ly/moe-batch-effect}
CODE=${CODE:-$BASE/src/moe-sparse-adaptation}
FULL=${FULL:-$BASE/data/rxrx1-full}
RESULT=${RESULT:-$BASE/results/rxrx1_calibration_20260817}
PYTHON=${PYTHON:-/hai/scratch/l1ly/experimental-ttl/env/bin/python}
PYBIN=$(dirname "$PYTHON")
EXTRA_SITE=${EXTRA_SITE:-$BASE/env-extra}
COMMIT=$(git -C "$CODE" rev-parse HEAD)
mkdir -p "$RESULT/logs"

EXPORT_BASE="ALL,CODE=$CODE,FULL=$FULL,PYTHON=$PYTHON,PYBIN=$PYBIN,EXTRA_SITE=$EXTRA_SITE,EXPECTED_COMMIT=$COMMIT,RAW_ROOT=$FULL/rxrx1,RESULT_ROOT=$RESULT"

PREP=$(sbatch --parsable -J rx1_manifest -p yejin -A yejin -t 02:00:00 \
  -c 16 --mem=64G --dependency="afterok:$STAGE_JOB" \
  -o "$RESULT/logs/prep-%j.out" -e "$RESULT/logs/prep-%j.err" \
  --export="$EXPORT_BASE" "$CODE/scripts/haic/prepare_rxrx1_calibration.sh")

OFFICIAL=$(sbatch --parsable -J rx1_dense_official -p yejin -A yejin \
  --gres=gpu:h200:8 --cpus-per-task=96 --mem=0 -t 1-00:00:00 \
  --dependency="afterok:$PREP" --requeue --signal=B:USR1@180 \
  -o "$RESULT/logs/official-%j.out" -e "$RESULT/logs/official-%j.err" \
  --export="$EXPORT_BASE,LANE=official" "$CODE/scripts/haic/run_rxrx1_calibration_job.sh")

CROSSFIT=$(sbatch --parsable -J rx1_huvec_xfit -p yejin -A yejin \
  --array=0-5%1 --gres=gpu:h200:4 --cpus-per-task=48 --mem=0 -t 1-00:00:00 \
  --dependency="afterok:$PREP" --requeue --signal=B:USR1@180 \
  -o "$RESULT/logs/crossfit-%A_%a.out" -e "$RESULT/logs/crossfit-%A_%a.err" \
  --export="$EXPORT_BASE,LANE=crossfit" "$CODE/scripts/haic/run_rxrx1_calibration_job.sh")

printf 'prep=%s\nofficial=%s\ncrossfit=%s\n' "$PREP" "$OFFICIAL" "$CROSSFIT"
