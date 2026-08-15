#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-$HOME/orcd/scratch/moe-batch-effect/worktrees/huvec-mae}
ENV_DIR=${ENV_DIR:-$HOME/orcd/scratch/moe-batch-effect/envs/huvec-mae-py311}

module purge
module load miniforge/24.3.0-0

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  conda create --yes --prefix "$ENV_DIR" python=3.11 pip
fi

"$ENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$ENV_DIR/bin/python" -m pip install \
  torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
"$ENV_DIR/bin/python" -m pip install --editable "$REPO[test]"
"$ENV_DIR/bin/python" - <<'PY'
import json
import torch
import torchvision
print(json.dumps({
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "torch_cuda": torch.version.cuda,
}, sort_keys=True))
PY
touch "$ENV_DIR/READY"
