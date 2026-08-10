#!/usr/bin/env python
"""Matched 2x4 RxRx1 architecture-by-training-environment scaling wave."""
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

from aggregate_rxrx1_domain_scaling import render_report
from moe_shift.capacity.naming import run_id_from
from moe_shift.data.rxrx1_scaling import (
    DESIGN_SEED,
    audit_environment_subset,
    full_environment_subset,
    quarter_environment_subset,
)
from moe_shift.utils import gpulease
from moe_shift.utils.config import apply_overrides, load_config


CONFIG = "configs/ccas_rxrx1_cell_dino_native.yaml"
CAMPAIGN = "rxrx1_domain_scaling30_20260810"
DEFAULT_ROOT = Path(
    "/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/"
    "substrate_rxrx1/cell_dino_cp5/rxrx1_domain_scaling30_20260810"
)
WANDB_GROUP = "rxrx1-cell-dino-domain-scaling30-20260810"
HF_PREFIX = "rxrx1/cell_dino_cp5/rxrx1_domain_scaling30_20260810"
SEED = 5


def _common(label):
    return [
        f"seed={SEED}", "stage=3", "model.routing_estimator=selected_st",
        "model.routing_unit=token", "model.geometry=cosine",
        "model.pressure=canonical", "model.balance=global",
        "model.freeze_backbone=false", "model.unfreeze_last_n_blocks=0",
        "model.sym_break_moe=0.0", "model.feature_stat_mix_prob=0.0",
        "model.router_frozen=false", "train.objective=erm",
        "train.cross_experiment_pairs=false", "train.epochs=30",
        "train.milestone_epochs=[5,10,20,30]", "train.save_checkpoint_epochs=[30]",
        "train.warmup_epochs=3", "train.llrd=1.0", "train.batch_size=64",
        "train.optim.lr=1.0e-4", "train.optim.weight_decay=0.05",
        "train.label_smoothing=0.0", "model.drop_path=0.1",
        "losses.balance_w=1.0e-2", "losses.zloss_w=1.0e-3",
        "losses.cross_experiment_contrastive_w=0.0",
        "analysis.run_mechanism=false", "analysis.record_train_accuracy=true",
        f"run_tag={CAMPAIGN}_{label}",
    ]


def _environment_override(environments):
    return "train.environment_subset=[" + ",".join(map(str, environments)) + "]"


def wave_rows(config=CONFIG):
    scales = (("quarter", quarter_environment_subset()), ("full", None))
    arms = (
        ("original", ["model.variant=original"]),
        ("dense_E4_late2", ["model.variant=dense_wide", "model.n_experts=4",
                            "model.ffn_block_indices=[10,11]"]),
        ("replace_E4k2_late2", ["model.variant=moe", "model.n_experts=4",
                                "model.top_k=2", "model.ffn_block_indices=[10,11]"]),
        ("shared_E3k1_late2", ["model.variant=shared_moe", "model.n_experts=3",
                               "model.top_k=1", "model.ffn_block_indices=[10,11]"]),
    )
    rows = []
    for scale, environments in scales:
        for arm, intervention in arms:
            label = f"{scale}_{arm}"
            overrides = [*_common(label), *intervention]
            if environments is not None:
                overrides.append(_environment_override(environments))
            cfg = apply_overrides(load_config(config), overrides)
            rows.append((label, scale, arm, environments, overrides, run_id_from(cfg), cfg))
    if len({row[5] for row in rows}) != len(rows):
        raise ValueError("RxRx1 domain-scaling run identities collide")
    return rows


def sharded_rows(rows, shard_index, num_shards):
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("require 0 <= shard_index < num_shards")
    return [row for index, row in enumerate(rows) if index % num_shards == shard_index]


def _dataset_audit(config):
    try:
        from wilds import get_dataset
        cfg = load_config(config)
        dataset = get_dataset(dataset="rxrx1", root_dir=cfg["data_root"], download=False)
    except Exception as error:
        return {"unavailable": f"{type(error).__name__}: {error}"}
    audits = {
        "quarter": audit_environment_subset(dataset, quarter_environment_subset()),
        "full": audit_environment_subset(dataset, full_environment_subset()),
    }
    if audits["quarter"]["n_classes_observed"] != audits["full"]["n_classes_observed"]:
        raise ValueError("quarter/full subsets do not preserve identical label coverage")
    if set(audits["quarter"]["cell_environment_counts"]) != {0, 1, 2, 3}:
        raise ValueError("quarter subset does not preserve all four cell types")
    return audits


def write_manifest(out, rows, config=CONFIG):
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "campaign": CAMPAIGN,
        "config": config,
        "seed": SEED,
        "design_seed": DESIGN_SEED,
        "axis": "number_of_independent_training_experiments",
        "headline_metrics": ["acc_heldout", "worst_env_heldout", "acc_within"],
        "fixed_controls": [
            "all 1139 perturbation labels represented at both scales",
            "all four cell types represented at both scales",
            "identical OOD validation/test and ID evaluation sets",
            "identical preprocessing, optimizer, epochs, checkpoint rule, and seed",
        ],
        "dataset_audit": _dataset_audit(config),
        "questions": [
            "Does shared-residual MoE improve relative to dense as independent training batches increase?",
            "Is any shared-MoE advantage due to retaining the dense path rather than sparse replacement?",
            "Do tail-batch gains scale differently from mean held-out-batch accuracy?",
        ],
        "runs": [
            {
                "label": label, "scale": scale, "arm": arm, "seed": SEED,
                "run_id": run_id, "environment_subset": list(environments or ()),
                "overrides": overrides, "variant": cfg["model"]["variant"],
            }
            for label, scale, arm, environments, overrides, run_id, cfg in rows
        ],
    }
    path = out / "wave_manifest.json"
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
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
        raise RuntimeError("tracking unavailable (" + ", ".join(missing) +
                           "); pass --allow-untracked for local-first execution")
    if missing:
        env["WANDB_MODE"] = "offline"
        print(f"[tracking] local-first mode; missing: {', '.join(missing)}", flush=True)
    env["WANDB_GROUP"] = WANDB_GROUP
    env["WANDB_JOB_TYPE"] = "rxrx1_domain_scaling30"
    env["WANDB_TAGS"] = "rxrx1,cell-dino,domain-scaling,shared-residual,matched-controls,stage3"
    env["CCAS_HF_PREFIX"] = HF_PREFIX
    return env


def command_for(row, config, out):
    return [sys.executable, "scripts/run_ccas.py", "--config", config,
            "--results-dir", str(out), "--override", *row[4]]


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
    write_manifest(out, all_rows, args.config)
    if args.status:
        print(render_report(out))
        return
    rows = sharded_rows(all_rows, args.shard_index, args.num_shards)
    pending = [row for row in rows if not (out / f"{row[5]}.json").exists()]
    print(render_report(out), flush=True)
    print(f"shard {args.shard_index}/{args.num_shards}: {len(rows)} planned, "
          f"{len(pending)} pending on GPUs {args.gpus}", flush=True)
    if args.dry_run:
        for row in rows:
            print(f"  {row[0]}: {row[5]}")
        return

    slots = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if args.max_concurrent > len(slots):
        raise ValueError("max-concurrent cannot exceed visible GPU slots")
    base_env = tracking_environment(require_tracking=not args.allow_untracked)
    running = {}
    while pending or running:
        while pending and len(running) < args.max_concurrent:
            gpu = gpulease.acquire_any(slots)
            if gpu is None:
                break
            row = pending.pop(0)
            label, run_id = row[0], row[5]
            env = dict(base_env, CUDA_VISIBLE_DEVICES=gpu)
            log_handle = open(out / f"{run_id}.log", "a")
            process = subprocess.Popen(command_for(row, args.config, out), env=env,
                                       stdout=log_handle, stderr=subprocess.STDOUT)
            gpulease.adopt(gpu, process.pid)
            running[gpu] = (process, run_id, label, log_handle)
            print(f"[start] gpu={gpu} pid={process.pid} {label} {run_id}", flush=True)
        for gpu in list(running):
            process, run_id, label, log_handle = running[gpu]
            if process.poll() is not None:
                log_handle.close()
                gpulease.release(gpu, pid=process.pid)
                print(f"[exit] gpu={gpu} rc={process.returncode} {label} {run_id}", flush=True)
                print(render_report(out), flush=True)
                del running[gpu]
        if pending or running:
            time.sleep(10)


if __name__ == "__main__":
    main()
