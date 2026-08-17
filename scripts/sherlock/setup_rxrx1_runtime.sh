#!/usr/bin/env bash
set -euo pipefail

: "${CODE:?set CODE to the frozen repository checkout}"
: "${ENV:?set ENV to the persistent virtual environment}"

module load py-pytorch/2.4.1_py312 py-torchvision/0.19.1_py312 \
  py-timm/1.0.12_py312 py-pandas/2.2.1_py312 \
  py-pyarrow/18.1.0_py312 py-scikit-learn/1.5.1_py312

if [[ ! -x "$ENV/bin/python" ]]; then
  python3.12 -m venv --system-site-packages "$ENV"
fi

"$ENV/bin/python" -m pip install --no-deps -e "$CODE"

"$ENV/bin/python" - <<'PY'
import pandas
import pyarrow
import timm
import torch
import sklearn
import torchvision
from PIL import Image
from torchvision.models import densenet161

model = densenet161(weights=None, memory_efficient=True)

print({
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "pandas": pandas.__version__,
    "pyarrow": pyarrow.__version__,
    "timm": timm.__version__,
    "sklearn": sklearn.__version__,
    "pillow": Image.__version__,
    "densenet_features": model.classifier.in_features,
})
assert torch.distributed.is_available()
PY
