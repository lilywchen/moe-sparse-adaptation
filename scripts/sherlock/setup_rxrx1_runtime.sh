#!/usr/bin/env bash
set -euo pipefail

: "${CODE:?set CODE to the frozen repository checkout}"
: "${ENV:?set ENV to the persistent virtual environment}"

module load py-pytorch/2.4.1_py312

if [[ ! -x "$ENV/bin/python" ]]; then
  python3.12 -m venv --system-site-packages "$ENV"
fi

"$ENV/bin/python" -m pip install --upgrade pip
"$ENV/bin/python" -m pip install --only-binary=:all: \
  pandas==2.2.3 pyarrow==17.0.0 pillow==10.4.0 \
  torchvision==0.19.0 timm==1.0.19 scikit-learn==1.5.2 \
  pyyaml==6.0.2
"$ENV/bin/python" -m pip install --no-deps -e "$CODE"

"$ENV/bin/python" - <<'PY'
import pandas
import pyarrow
import timm
import torch
import torchvision
from PIL import Image

print({
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "pandas": pandas.__version__,
    "pyarrow": pyarrow.__version__,
    "timm": timm.__version__,
    "pillow": Image.__version__,
})
assert hasattr(torch.distributed, "run") or torch.distributed.is_available()
PY

