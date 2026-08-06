#!/usr/bin/env python
"""Eight-GPU corrected-router and sparse-depth screen for Cell-DINO on RxRx1.

Screen phase (eight seed-0 arms, one per GPU) jointly tests:
  * corrected learned routing versus frozen routing at one and two late FFNs;
  * sparse versus equal-total-parameter dense capacity at two late FFNs;
  * sparse depth at L={1,2,4,12}, with the original model as the L=0 anchor.

Confirmation phase takes a winning block list and fills eight GPUs with the exact
original/dense/learned/frozen quartet at seeds 1 and 2. OOD test remains sealed throughout.
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
CAMPAIGN = "routerfix_depth30_20260806"
DEFAULT_ROOT = Path(
    "/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/"
    "substrate_rxrx1/cell_dino_cp5/routerfix_depth30_20260806"
)


def _blocks_override(blocks):
    return "model.ffn_block_indices=[" + ",".join(str(i) for i in blocks) + "]"


def _common(seed, run_tag):
    return [
        f"seed={seed}", "stage=1", "model.n_experts=8", "model.top_k=1",
        "model.routing_estimator=selected_st", "model.routing_unit=token",
        "model.geometry=cosine", "model.pressure=canonical", "model.balance=global",
        "model.freeze_backbone=false", "model.unfreeze_last_n_blocks=0",
        "model.sym_break_moe=0.0", "train.objective=erm", "train.epochs=30",
        "train.milestone_epochs=[5,10,20,30]",
        "train.save_checkpoint_epochs=[10,30]", "train.warmup_epochs=3",
        "train.llrd=1.0", "train.batch_size=64", "train.optim.lr=1.0e-4",
        "train.optim.weight_decay=0.05", "model.drop_path=0.1",
        "losses.balance_w=1.0e-2", "losses.zloss_w=1.0e-3",
        "analysis.run_mechanism=true", "analysis.record_train_accuracy=false",
        f"run_tag={run_tag}",
    ]


def _row(config, label, variant, blocks, seed, run_tag):
    overrides = [*_common(seed, run_tag), f"model.variant={variant}"]
    if blocks is not None:
        overrides.append(_blocks_override(blocks))
    cfg = apply_overrides(load_config(config), overrides)
    return label, overrides, run_id_from(cfg)


def screen_cells(config=CONFIG):
    """Exactly eight nonredundant arms for the first parallel wave."""
    tag = CAMPAIGN
    specs = [
        ("original", "original", None),
        ("dense_last2", "dense_wide", (10, 11)),
        ("learned_last1", "moe", (11,)),
        ("frozen_last1", "moe_frozen", (11,)),
        ("learned_last2", "moe", (10, 11)),
        ("frozen_last2", "moe_frozen", (10, 11)),
        ("learned_last4", "moe", (8, 9, 10, 11)),
        ("learned_all12", "moe", tuple(range(12))),
    ]
    return [_row(config, label, variant, blocks, seed=0, run_tag=tag)
            for label, variant, blocks in specs]


def confirmation_cells(blocks, config=CONFIG):
    """Two fresh seeds x the four-arm causal/capacity quartet = eight GPUs."""
    blocks = tuple(int(i) for i in blocks)
    if not blocks:
        raise ValueError("confirmation requires at least one block index")
    block_label = "-".join(str(i) for i in blocks)
    tag = f"routerfix_confirm30_blocks{block_label}_20260806"
    rows = []
    for seed in (1, 2):
        for label, variant, selected in (
            ("original", "original", None),
            ("dense", "dense_wide", blocks),
            ("learned", "moe", blocks),
            ("frozen", "moe_frozen", blocks),
        ):
            rows.append(_row(
                config, f"{label}_s{seed}", variant, selected, seed=seed, run_tag=tag))
    return rows


def sharded_rows(rows, shard_index, num_shards):
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("require 0 <= shard_index < num_shards")
    return [row for index, row in enumerate(rows) if index % num_shards == shard_index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--phase", choices=("screen", "confirm"), default="screen")
    parser.add_argument(
        "--confirm-blocks", default=None,
        help="comma-separated zero-based blocks; required for --phase confirm",
    )
    parser.add_argument("--result-root", default=None)
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"))
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.phase == "screen":
        rows = screen_cells(args.config)
        default_out = DEFAULT_ROOT / "screen"
        group = "rxrx1-cell-dino-routerfix-depth30-screen-20260806"
    else:
        if not args.confirm_blocks:
            parser.error("--confirm-blocks is required for --phase confirm")
        blocks = tuple(int(i) for i in args.confirm_blocks.split(","))
        rows = confirmation_cells(blocks, args.config)
        block_label = "-".join(str(i) for i in blocks)
        default_out = DEFAULT_ROOT / f"confirm_blocks{block_label}"
        group = f"rxrx1-cell-dino-routerfix-confirm30-blocks{block_label}-20260806"

    out = Path(args.result_root) if args.result_root else default_out
    rows = sharded_rows(rows, args.shard_index, args.num_shards)
    pending = [row for row in rows if not (out / f"{row[2]}.json").exists()]
    if args.dry_run:
        print(
            f"{args.phase} shard {args.shard_index}/{args.num_shards}: "
            f"{len(rows)} planned, {len(pending)} pending -> {out}"
        )
        for label, _, run_id in rows:
            print(f"  {label}: {run_id}")
        return

    out.mkdir(parents=True, exist_ok=True)
    slots = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    running = {}
    while pending or running:
        while pending and len(running) < args.max_concurrent:
            gpu = gpulease.acquire_any(slots)
            if gpu is None:
                break
            label, overrides, run_id = pending.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            env["WANDB_GROUP"] = group
            env["WANDB_JOB_TYPE"] = f"rxrx1_routerfix_depth30_{args.phase}"
            env["WANDB_TAGS"] = (
                "rxrx1,cell-dino,selected-st,depth-screen,exploratory,ood-test-blind"
            )
            # Training artifacts stay on persistent storage; publishing happens after validation.
            env["HF_TOKEN"] = ""
            log_handle = open(out / f"{run_id}.log", "a")
            command = [
                sys.executable, "scripts/run_ccas.py", "--config", args.config,
                "--results-dir", str(out), "--override", *overrides,
            ]
            process = subprocess.Popen(
                command, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
            gpulease.adopt(gpu, process.pid)
            running[gpu] = (process, run_id, label, log_handle)
            print(
                f"[start] shard={args.shard_index} gpu={gpu} pid={process.pid} "
                f"{label} {run_id}", flush=True)

        for gpu in list(running):
            process, run_id, label, log_handle = running[gpu]
            if process.poll() is not None:
                log_handle.close()
                gpulease.release(gpu, pid=process.pid)
                print(
                    f"[exit] gpu={gpu} rc={process.returncode} {label} {run_id}", flush=True)
                del running[gpu]
        if pending or running:
            time.sleep(10)


if __name__ == "__main__":
    main()
