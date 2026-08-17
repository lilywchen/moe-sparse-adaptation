#!/usr/bin/env bash
set -euo pipefail

: "${CODE:?set CODE}"
: "${DATA_ROOT:?set DATA_ROOT}"
: "${ENV:?set ENV}"

module load py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 \
  py-timm/1.0.12_py312 py-pandas/2.2.1_py312 \
  py-pyarrow/18.1.0_py312 py-scikit-learn/1.5.1_py312
PYTHON="$ENV/bin/python"
test -x "$PYTHON"
test -f "$DATA_ROOT/EXTRACTION_COMPLETE.json"
test -f "$DATA_ROOT/rxrx1/metadata.csv"

"$PYTHON" "$CODE/scripts/build_rxrx1_full_manifest.py" \
  --metadata-csv "$DATA_ROOT/rxrx1/metadata.csv" \
  --raw-root "$DATA_ROOT/rxrx1" \
  --output-root "$DATA_ROOT/manifests"
"$PYTHON" "$CODE/scripts/freeze_rxrx1_huvec_crossfits.py" \
  --all-sites "$DATA_ROOT/manifests/all_sites.parquet" \
  --output-root "$DATA_ROOT/huvec_crossfits"

test -s "$DATA_ROOT/manifests/full_manifest.summary.json"
test -s "$DATA_ROOT/huvec_crossfits/huvec_crossfit_registry.json"
printf '[ready] frozen manifest and HUVEC cross-fits\n'
