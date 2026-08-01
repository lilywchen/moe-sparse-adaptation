#!/usr/bin/env python
"""Bounded Cell-DINO substrate-strength screen for RxRx1.

The first ten-epoch native-CP5 run established a learnable batch-transfer regime but remained
well below the completed local WILDS ResNet reference.  This launcher uses one H100 per arm to
separate the most plausible optimization explanations before sparse-capacity work is reconsidered.

The screen is intentionally finite, seed-0, OOD-validation-only, and shardable across independent
two-GPU SciServer containers.  Shards are disjoint by construction and the result namespace is
separate from the completed kill campaign.
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


# All arms share seed, data, preprocessing, model, 30 epochs, and three warmup epochs.  Each tag
# names its sole intervention relative to the 1e-4 / wd=.05 / uniform-LR / batch-64 anchor.
SCREEN_SPECS = [
    ("lr3e-5", ["train.optim.lr=3.0e-5"]),
    ("lr6e-5", ["train.optim.lr=6.0e-5"]),
    ("anchor_lr1e-4", ["train.optim.lr=1.0e-4"]),
    ("lr2e-4", ["train.optim.lr=2.0e-4"]),
    ("llrd0.85", ["train.optim.lr=1.0e-4", "train.llrd=0.85"]),
    ("llrd0.95", ["train.optim.lr=1.0e-4", "train.llrd=0.95"]),
    ("wd0.01", ["train.optim.lr=1.0e-4", "train.optim.weight_decay=0.01"]),
    ("wd0.10", ["train.optim.lr=1.0e-4", "train.optim.weight_decay=0.10"]),
    ("drop_path0", ["train.optim.lr=1.0e-4", "model.drop_path=0.0"]),
    ("batch128_lr2e-4", ["train.batch_size=128", "train.optim.lr=2.0e-4"]),
]


def screen_rows(config=CONFIG):
    rows = []
    common = [
        "model.variant=original", "model.freeze_backbone=false", "train.epochs=30",
        "train.warmup_epochs=3", "train.llrd=1.0", "train.batch_size=64",
        "train.optim.weight_decay=0.05", "model.drop_path=0.1",
        "analysis.run_mechanism=false", "seed=0",
    ]
    for tag, intervention in SCREEN_SPECS:
        run_tag = f"strength30_{tag}"
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

    out = Path(args.result_root) if args.result_root else RESULT_ROOT / "screen30"
    out.mkdir(parents=True, exist_ok=True)
    rows = sharded_rows(screen_rows(args.config), args.shard_index, args.num_shards)
    pending = [row for row in rows if not (out / f"{row[2]}.json").exists()]
    if args.dry_run:
        print(f"RxRx1 strength shard {args.shard_index}/{args.num_shards}: "
              f"{len(rows)} planned, {len(pending)} pending -> {out}")
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
