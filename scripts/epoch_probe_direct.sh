#!/bin/bash
# Direct 90-epoch rxrx1 epoch-budget probe.
# The earlier queued version (epoch_probe_when_free.sh) died inside its "wait for
# idle GPUs" sleep loop at 18:11 without ever reaching [go].  GPUs are free now,
# so we skip the wait entirely and detach with setsid so a kernel death cannot
# reap the jobs.  Idempotent: skips any cell whose result JSON already exists.
set -u
REPO=/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation
OUT=$REPO/hpo/rxrx1/epoch_probe
LOG=$REPO/logs/epoch_probe.launch.log
source /home/idies/miniconda3/etc/profile.d/conda.sh
conda activate /home/idies/workspace/Storage/lchen5/persistent/envs/moe
cd $REPO || exit 1
mkdir -p $OUT
echo "[go-direct] $(date -u +%FT%TZ) gpus free, launching without wait loop" >> $LOG

launch () {  # $1 gpu  $2 llrd
  tag=epochprobe_ep90_lr1e-04_llrd$2
  rid=rxrx1_original_ep90_s0_$tag
  if [ -f "$OUT/$rid.json" ]; then echo "[skip] $rid" >> $LOG; return; fi
  echo "[start] $(date -u +%FT%TZ) gpu=$1 $rid" >> $LOG
  CUDA_VISIBLE_DEVICES=$1 python scripts/run_ccas.py \
    --config configs/ccas_rxrx1.yaml --results-dir $OUT \
    --override seed=0 model.variant=original model.pressure=canonical \
      train.epochs=90 train.optim.lr=0.0001 train.llrd=$2 \
      train.warmup_epochs=5 train.num_workers=6 run_tag=$tag \
    > $OUT/$rid.log 2>&1
  echo "[exit] $(date -u +%FT%TZ) $rid rc=$?" >> $LOG
}

launch 0 0.70 &
launch 1 0.85 &
wait
echo "[done] $(date -u +%FT%TZ)" >> $LOG
