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

PREP=$(sbatch --parsable -J rx1_manifest -p yejin -A yejin -t 02:00:00 \
  -c 16 --mem=64G --dependency="afterok:$STAGE_JOB" \
  -o "$RESULT/logs/prep-%j.out" -e "$RESULT/logs/prep-%j.err" \
  --wrap="bash -lc '"'"'set -euo pipefail; $PYTHON $CODE/scripts/build_rxrx1_full_manifest.py --metadata-csv $FULL/rxrx1/metadata.csv --raw-root $FULL/rxrx1 --output-root $FULL/manifests; $PYTHON $CODE/scripts/freeze_rxrx1_huvec_crossfits.py --all-sites $FULL/manifests/all_sites.parquet --output-root $FULL/huvec_crossfits'"'"'")

COMMON="export PATH=$PYBIN:\$PATH PYTHONPATH=$EXTRA_SITE; CODE=$CODE EXPECTED_COMMIT=$COMMIT RAW_ROOT=$FULL/rxrx1 RESULT_ROOT=$RESULT"
OFFICIAL=$(sbatch --parsable -J rx1_dense_official -p yejin -A yejin \
  --gres=gpu:h200:8 --cpus-per-task=96 --mem=0 -t 1-00:00:00 \
  --dependency="afterok:$PREP" --requeue --signal=B:USR1@180 \
  -o "$RESULT/logs/official-%j.out" -e "$RESULT/logs/official-%j.err" \
  --wrap="bash -lc '"'"'$COMMON MANIFEST=$FULL/manifests/all_sites.parquet RUN_NAME=official_densenet161 MODEL=densenet161 NPROC=8 PER_GPU_BATCH=16 WORKERS=10 EXTRA_ARGS=--adabn bash $CODE/scripts/multicluster/run_rxrx1_calibration.sh'"'"'")

CROSSFIT=$(sbatch --parsable -J rx1_huvec_xfit -p yejin -A yejin \
  --array=0-5%1 --gres=gpu:h200:4 --cpus-per-task=48 --mem=0 -t 1-00:00:00 \
  --dependency="afterok:$PREP" --requeue --signal=B:USR1@180 \
  -o "$RESULT/logs/crossfit-%A_%a.out" -e "$RESULT/logs/crossfit-%A_%a.err" \
  --wrap="bash -lc '"'"'$COMMON MANIFEST=$FULL/huvec_crossfits/huvec_crossfit\${SLURM_ARRAY_TASK_ID}_h16.parquet RUN_NAME=huvec_crossfit\${SLURM_ARRAY_TASK_ID}_h16_densenet MODEL=densenet161 NPROC=4 PER_GPU_BATCH=16 WORKERS=10 SPLIT=custom EXTRA_ARGS= bash $CODE/scripts/multicluster/run_rxrx1_calibration.sh'"'"'")

printf 'prep=%s\nofficial=%s\ncrossfit=%s\n' "$PREP" "$OFFICIAL" "$CROSSFIT"
