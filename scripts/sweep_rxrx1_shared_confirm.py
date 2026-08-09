#!/usr/bin/env python
"""Fresh-seed confirmation of shared residual MoE on Cell-DINO/RxRx1.

This is the smallest wave that can adjudicate the current seed-0 result.  At each
of two fresh seeds it compares the untouched encoder, an equal-total dense
expansion, traditional replacement MoE, and the leading shared residual MoE.
The three expanded arms use four FFN copies worth of total capacity and two FFN
paths worth of active compute at blocks 10--11:

* dense E4: one four-times-wide shared path;
* replacement E4/top-2: two of four expert paths;
* shared E3/top-1: the pretrained shared path plus one of three residual paths.

OOD validation is the decision split.  OOD test is a fixed-arm descriptive
readout because the entire eight-run registry is declared before launch.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from aggregate_shared_confirm import render_report
from moe_shift.capacity.naming import run_id_from
from moe_shift.utils import gpulease
from moe_shift.utils.config import apply_overrides, load_config


CONFIG = "configs/ccas_rxrx1_cell_dino_native.yaml"
CAMPAIGN = "shared_confirm30_20260809"
DEFAULT_ROOT = Path(
    "/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/"
    "substrate_rxrx1/cell_dino_cp5/shared_confirm30_20260809"
)
WANDB_GROUP = "rxrx1-cell-dino-shared-confirm30-20260809"
HF_PREFIX = "rxrx1/cell_dino_cp5/shared_confirm30_20260809"
SEEDS = (1, 2)
LATE2 = (10, 11)


def _blocks(indices):
    return "model.ffn_block_indices=[" + ",".join(str(index) for index in indices) + "]"


def _common(seed, label):
    return [
        f"seed={seed}", "stage=3",
        "model.routing_estimator=selected_st",
        "model.routing_unit=token", "model.geometry=cosine",
        "model.pressure=canonical", "model.balance=global",
        "model.freeze_backbone=false", "model.unfreeze_last_n_blocks=0",
        "model.sym_break_moe=0.0", "model.feature_stat_mix_prob=0.0",
        "train.objective=erm", "train.cross_experiment_pairs=false",
        "train.epochs=30", "train.milestone_epochs=[5,10,20,30]",
        "train.save_checkpoint_epochs=[30]", "train.warmup_epochs=3",
        "train.llrd=1.0", "train.batch_size=64", "train.optim.lr=1.0e-4",
        "train.optim.weight_decay=0.05", "train.label_smoothing=0.0",
        "model.drop_path=0.1", "losses.balance_w=1.0e-2",
        "losses.zloss_w=1.0e-3", "losses.cross_experiment_contrastive_w=0.0",
        "analysis.run_mechanism=true", "analysis.record_train_accuracy=true",
        f"run_tag={CAMPAIGN}_{label}_s{seed}",
    ]


def wave_rows(config=CONFIG):
    specs = (
        ("original", ["model.variant=original"]),
        ("dense_E4_late2", [
            "model.variant=dense_wide", "model.n_experts=4", "model.top_k=2",
            _blocks(LATE2),
        ]),
        ("replace_E4k2_late2", [
            "model.variant=moe", "model.n_experts=4", "model.top_k=2",
            _blocks(LATE2),
        ]),
        ("shared_E3k1_late2", [
            "model.variant=shared_moe", "model.n_experts=3", "model.top_k=1",
            _blocks(LATE2),
        ]),
    )
    rows = []
    for seed in SEEDS:
        for label, intervention in specs:
            overrides = [*_common(seed, label), *intervention]
            cfg = apply_overrides(load_config(config), overrides)
            rows.append((f"{label}_s{seed}", label, seed, overrides, run_id_from(cfg), cfg))
    run_ids = [row[4] for row in rows]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("shared-confirm arms collide on run identity")
    return rows


def sharded_rows(rows, shard_index, num_shards):
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("require 0 <= shard_index < num_shards")
    return [row for index, row in enumerate(rows) if index % num_shards == shard_index]


def write_manifest(out, rows):
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "campaign": CAMPAIGN,
        "config": CONFIG,
        "selection_split": "ood_val",
        "test_readout": "all_predeclared_arms",
        "seeds": list(SEEDS),
        "placement_fixed": list(LATE2),
        "label_smoothing_fixed": 0.0,
        "fairness": (
            "dense_E4, replacement_E4k2, and shared_E3k1 have approximately equal total "
            "capacity and two FFN paths of active compute at each converted block"
        ),
        "primary_contrasts": [
            "shared_E3k1_late2 - dense_E4_late2",
            "shared_E3k1_late2 - replace_E4k2_late2",
            "dense_E4_late2 - original",
        ],
        "runs": [
            {
                "label": display, "arm": arm, "seed": seed, "run_id": run_id,
                "overrides": overrides, "variant": cfg["model"]["variant"],
                "n_experts": cfg["model"]["n_experts"], "top_k": cfg["model"]["top_k"],
            }
            for display, arm, seed, overrides, run_id, cfg in rows
        ],
    }
    path = out / "wave_manifest.json"
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    os.replace(temporary, path)
    return payload


def tracking_environment(require_tracking=True):
    env = dict(os.environ)
    if not env.get("HF_TOKEN"):
        try:
            from huggingface_hub import get_token
            env["HF_TOKEN"] = get_token() or ""
        except Exception:
            pass
    if not env.get("WANDB_API_KEY"):
        try:
            import wandb
            env["WANDB_API_KEY"] = wandb.api.api_key or ""
        except Exception:
            pass
    missing = [name for name in ("WANDB_API_KEY", "HF_TOKEN", "CCAS_HF_REPO")
               if not env.get(name)]
    if missing and require_tracking:
        raise RuntimeError(
            "tracking is unavailable (" + ", ".join(missing)
            + "); pass --allow-untracked to preserve local artifacts and offline W&B")
    if missing:
        env["WANDB_MODE"] = "offline"
        print(f"[tracking] local-first mode; missing: {', '.join(missing)}", flush=True)
    env["WANDB_GROUP"] = WANDB_GROUP
    env["WANDB_JOB_TYPE"] = "rxrx1_shared_confirm30"
    env["WANDB_TAGS"] = "rxrx1,cell-dino,shared-residual,confirmation,fresh-seeds,stage3"
    env["CCAS_HF_PREFIX"] = HF_PREFIX
    return env


def command_for(row, config, out):
    _display, _arm, _seed, overrides, _run_id, _cfg = row
    return [sys.executable, "scripts/run_ccas.py", "--config", config,
            "--results-dir", str(out), "--override", *overrides]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--result-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"))
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--allow-untracked", action="store_true")
    args = parser.parse_args()

    out = Path(args.result_root).expanduser().resolve()
    all_rows = wave_rows(args.config)
    write_manifest(out, all_rows)
    if args.status:
        print(render_report(out))
        return

    rows = sharded_rows(all_rows, args.shard_index, args.num_shards)
    pending = [row for row in rows if not (out / f"{row[4]}.json").exists()]
    print(render_report(out), flush=True)
    print(f"shard {args.shard_index}/{args.num_shards}: {len(rows)} planned, "
          f"{len(pending)} pending on GPUs {args.gpus}", flush=True)
    if args.dry_run:
        for row in rows:
            print(f"  {row[0]}: {row[4]}")
        return

    slots = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if args.max_concurrent > len(slots):
        raise ValueError("max-concurrent cannot exceed the number of visible GPU slots")
    base_env = tracking_environment(require_tracking=not args.allow_untracked)
    running = {}
    while pending or running:
        while pending and len(running) < args.max_concurrent:
            gpu = gpulease.acquire_any(slots)
            if gpu is None:
                break
            row = pending.pop(0)
            display, _arm, _seed, _overrides, run_id, _cfg = row
            env = dict(base_env, CUDA_VISIBLE_DEVICES=gpu)
            log_handle = open(out / f"{run_id}.log", "a")
            process = subprocess.Popen(
                command_for(row, args.config, out), env=env,
                stdout=log_handle, stderr=subprocess.STDOUT)
            gpulease.adopt(gpu, process.pid)
            running[gpu] = (process, run_id, display, log_handle)
            print(f"[start] gpu={gpu} pid={process.pid} {display} {run_id}", flush=True)

        for gpu in list(running):
            process, run_id, display, log_handle = running[gpu]
            if process.poll() is not None:
                log_handle.close()
                gpulease.release(gpu, pid=process.pid)
                print(f"[exit] gpu={gpu} rc={process.returncode} {display} {run_id}", flush=True)
                print(render_report(out), flush=True)
                del running[gpu]
        if pending or running:
            time.sleep(10)


if __name__ == "__main__":
    main()
