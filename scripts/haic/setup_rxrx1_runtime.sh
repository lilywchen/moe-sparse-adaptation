#!/usr/bin/env bash
set -euo pipefail

: "${MAMBA:?}"
: "${MAMBA_ROOT:?}"
: "${RUNTIME:?}"
: "${EXTRA_SITE:?}"

"$MAMBA" create -y -r "$MAMBA_ROOT" -p "$RUNTIME" -c conda-forge python=3.10
PYTHONPATH="$EXTRA_SITE" "$RUNTIME/bin/python" - <<'PY'
import sys
import dateutil
import numpy
import pandas
import PIL
import pyarrow
import pytz
import torch
import torchvision

print(sys.version)
print({
    "numpy": numpy.__version__,
    "pandas": pandas.__version__,
    "pyarrow": pyarrow.__version__,
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
})
PY
