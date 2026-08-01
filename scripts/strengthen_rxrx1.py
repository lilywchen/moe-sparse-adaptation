#!/usr/bin/env python
"""Hypothesis-driven 90-epoch Cell-DINO matrix for RxRx1.

The first ten-epoch native-CP5 comparison may have ended before the pretrained transformer was
fully adapted.  A learning-rate grid would answer that question inefficiently and would say little
about *why* the model fails.  This matrix therefore runs one shared, benchmark-length schedule and
spends the parallel arms on distinct explanations: insufficient representation, destructive full
fine-tuning, explicit environment imbalance, batch invariance, routing granularity, routing
starvation, and shared versus conditional capacity.

All arms report train/ID/OOD-validation metrics at epochs 10, 30, 60, and 90.  Only the original
anchor stores all four model checkpoints; the other arms are independent scientific contrasts, not
duplicate epoch-budget runs.  The matrix is seed-0, OOD-test-blind, finite, and sharded across five
independent two-GPU containers.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe_shift.capacity.naming import run_id_from
from moe_shift.utils import gpulease
from moe_shift.utils.config import apply_overrides, load_config


CONFIG = "configs/ccas_rxrx1_cell_dino_native.yaml"
RESULT_ROOT = Path(os.environ.get(
    "MOE_RX_STRENGTH_RESULTS",
    "/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/"
    "substrate_rxrx1/cell_dino_cp5",
))
WANDB_GROUP = os.environ.get("MOE_RX_WANDB_GROUP", "rxrx1-cell-dino-hypothesis90-20260801")
HF_PREFIX = os.environ.get(
    "MOE_RX_HF_PREFIX", "rxrx1/cell_dino_cp5/hypothesis90_20260801"
)


# Each arm changes a scientific mechanism relative to ``original_anchor``.  These are deliberately
# not optimizer-tuning arms.  The top-2 arm changes active compute and is diagnostic rather than an
# exact-active-compute comparator; total parameters remain matched to the other MoE models.
SCREEN_SPECS = [
    ("original_anchor", ["model.variant=original",
                         "train.save_checkpoint_epochs=[10,30,60,90]"]),
    ("dense_wide", ["model.variant=dense_wide"]),
    ("moe_token_top1", ["model.variant=moe", "model.routing_unit=token",
                         "model.pressure=canonical", "model.top_k=1"]),
    ("moe_image_top1", ["model.variant=moe", "model.routing_unit=image",
                         "model.pressure=canonical", "model.top_k=1"]),
    ("moe_token_within_env", ["model.variant=moe", "model.routing_unit=token",
                               "model.pressure=route", "model.top_k=1"]),
    ("moe_token_top2", ["model.variant=moe", "model.routing_unit=token",
                         "model.pressure=canonical", "model.top_k=2"]),
    ("frozen_linear", ["model.variant=original", "model.freeze_backbone=true",
                        "train.optim.lr=1.0e-3"]),
    ("partial_last4", ["model.variant=original", "model.unfreeze_last_n_blocks=4"]),
    ("output_invariant", ["model.variant=original", "model.pressure=output",
                           "losses.invariance_w=0.1"]),
    ("environment_balanced", ["model.variant=original",
                               "train.objective=environment_balanced"]),
]


def screen_rows(config=CONFIG):
    rows = []
    common = [
        "model.variant=original", "model.freeze_backbone=false",
        "model.unfreeze_last_n_blocks=0", "model.pressure=canonical", "model.top_k=1",
        "train.objective=erm", "train.epochs=90", "train.milestone_epochs=[10,30,60,90]",
        "train.save_checkpoint_epochs=[]", "train.warmup_epochs=5", "train.llrd=1.0",
        "train.batch_size=64", "train.optim.lr=1.0e-4",
        "train.optim.weight_decay=0.05", "model.drop_path=0.1",
        "analysis.run_mechanism=false", "seed=0",
    ]
    for tag, intervention in SCREEN_SPECS:
        run_tag = f"hypothesis90_{tag}"
        overrides = [*common, *intervention, f"run_tag={run_tag}"]
        cfg = apply_overrides(load_config(config), overrides)
        rows.append((tag, overrides, run_id_from(cfg)))
    return rows


def sharded_rows(rows, shard_index, num_shards):
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("require 0 <= shard_index < num_shards")
    return [row for i, row in enumerate(rows) if i % num_shards == shard_index]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--result-root", default=None)
    ap.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"))
    ap.add_argument("--max-concurrent", type=int, default=2)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = Path(args.result_root) if args.result_root else RESULT_ROOT / "hypothesis90"
    out.mkdir(parents=True, exist_ok=True)
    rows = sharded_rows(screen_rows(args.config), args.shard_index, args.num_shards)
    pending = [row for row in rows if not (out / f"{row[2]}.json").exists()]
    if args.dry_run:
        print(f"RxRx1 hypothesis shard {args.shard_index}/{args.num_shards}: "
              f"{len(rows)} planned, {len(pending)} pending -> {out}")
        print(f"  W&B group: {WANDB_GROUP}; HF prefix: {HF_PREFIX}")
        for tag, _, rid in rows:
            print(f"  {tag}: {rid}")
        return

    slots = [x.strip() for x in args.gpus.split(",")]
    running = {}
    while pending or running:
        while pending and len(running) < args.max_concurrent:
            gpu = gpulease.acquire_any(slots)
            if gpu is None:
                break
            tag, overrides, rid = pending.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            env["WANDB_GROUP"] = WANDB_GROUP
            env["WANDB_JOB_TYPE"] = "rxrx1_hypothesis_matrix"
            env["WANDB_TAGS"] = "rxrx1,cell-dino,hypothesis90,ood-test-blind"
            env["CCAS_HF_PREFIX"] = HF_PREFIX
            log = open(out / f"{rid}.log", "a")
            cmd = [sys.executable, "scripts/run_ccas.py", "--config", args.config,
                   "--results-dir", str(out), "--override", *overrides]
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            gpulease.adopt(gpu, proc.pid)
            running[gpu] = (proc, rid, log)
            print(f"[start] shard={args.shard_index} gpu={gpu} pid={proc.pid} {tag} {rid}",
                  flush=True)
        for gpu in list(running):
            proc, rid, log = running[gpu]
            if proc.poll() is not None:
                log.close()
                gpulease.release(gpu, pid=proc.pid)
                print(f"[exit] {rid} rc={proc.returncode}", flush=True)
                del running[gpu]
        if pending or running:
            time.sleep(10)


if __name__ == "__main__":
    main()
