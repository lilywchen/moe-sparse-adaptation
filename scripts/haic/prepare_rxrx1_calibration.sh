#!/usr/bin/env bash
set -euo pipefail
: "${CODE:?}"
: "${FULL:?}"
: "${PYTHON:?}"
: "${EXTRA_SITE:?}"

export PYTHONPATH="$EXTRA_SITE${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" "$CODE/scripts/build_rxrx1_full_manifest.py" \
  --metadata-csv "$FULL/rxrx1/metadata.csv" --raw-root "$FULL/rxrx1" \
  --output-root "$FULL/manifests"
"$PYTHON" "$CODE/scripts/freeze_rxrx1_huvec_crossfits.py" \
  --all-sites "$FULL/manifests/all_sites.parquet" \
  --output-root "$FULL/huvec_crossfits"
