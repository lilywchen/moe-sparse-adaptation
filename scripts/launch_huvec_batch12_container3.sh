#!/usr/bin/env bash
set -euo pipefail
REPO=/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation-huvec-batch12
PY=/home/idies/workspace/Storage/lchen5/persistent/envs/moe/bin/python
cd "$REPO"
export CCAS_GPU_LOCK_DIR=/tmp/huvec_batch12_gpu_locks
exec "$PY" scripts/sweep_rxrx1_huvec_batch_effect.py --shard-index 0 --num-shards 2 --gpus 0,1 --max-concurrent 2
