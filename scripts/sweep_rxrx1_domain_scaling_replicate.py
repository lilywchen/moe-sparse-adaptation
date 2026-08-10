#!/usr/bin/env python
"""Seed-1/2 replication of the eight-experiment RxRx1 scaling endpoint."""
import argparse
import copy
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

from aggregate_rxrx1_domain_scaling_replicate import normalized_config, render_report
from moe_shift.capacity.naming import run_id_from
from moe_shift.data.rxrx1_scaling import audit_environment_subset, quarter_environment_subset
from moe_shift.utils import gpulease
from moe_shift.utils.config import apply_overrides, load_config
from sweep_rxrx1_shared_confirm import wave_rows as full_anchor_rows


CONFIG = "configs/ccas_rxrx1_cell_dino_native.yaml"
CAMPAIGN = "rxrx1_domain_scaling_replicate30_20260810"
DEFAULT_ROOT = Path(
    "/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/"
    "substrate_rxrx1/cell_dino_cp5/rxrx1_domain_scaling_replicate30_20260810"
)
FULL_ANCHOR_ROOT = Path(
    "/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/"
    "substrate_rxrx1/cell_dino_cp5/shared_confirm30_20260809"
)
WANDB_GROUP = "rxrx1-cell-dino-domain-scaling-replicate30-20260810"
HF_PREFIX = "rxrx1/cell_dino_cp5/rxrx1_domain_scaling_replicate30_20260810"
SEEDS = (1, 2)
ARMS = (
    ("original", ["model.variant=original"]),
    ("dense_E4_late2", ["model.variant=dense_wide", "model.n_experts=4",
                         "model.top_k=2", "model.ffn_block_indices=[10,11]"]),
    ("replace_E4k2_late2", ["model.variant=moe", "model.n_experts=4",
                             "model.top_k=2", "model.ffn_block_indices=[10,11]"]),
    ("shared_E3k1_late2", ["model.variant=shared_moe", "model.n_experts=3",
                            "model.top_k=1", "model.ffn_block_indices=[10,11]"]),
)


def _common(seed, label):
    # Intentionally identical to the completed seed-1/2 full-data anchor protocol.
    return [
        f"seed={seed}", "stage=3", "model.routing_estimator=selected_st",
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


def _environment_override(environments):
    return "train.environment_subset=[" + ",".join(map(str, environments)) + "]"


def wave_rows(config=CONFIG):
    environments = quarter_environment_subset()
    rows = []
    for seed in SEEDS:
        for arm, intervention in ARMS:
            label = f"quarter_{arm}_s{seed}"
            overrides = [*_common(seed, arm), *intervention,
                         _environment_override(environments)]
            cfg = apply_overrides(load_config(config), overrides)
            rows.append((label, arm, seed, environments, overrides, run_id_from(cfg), cfg))
    if len({row[5] for row in rows}) != len(rows):
        raise ValueError("domain-scaling replication run identities collide")
    validate_planned_pairs(rows, config)
    return rows


def validate_planned_pairs(rows, config=CONFIG):
    anchors = {(row[1], int(row[2])): row[5] for row in full_anchor_rows(config)}
    for _label, arm, seed, _environments, _overrides, _run_id, cfg in rows:
        anchor = anchors.get((arm, int(seed)))
        if anchor is None:
            raise ValueError(f"missing completed full anchor declaration for {arm}/seed{seed}")
        if normalized_config(cfg) != normalized_config(anchor):
            raise ValueError(f"planned quarter/full config drift for {arm}/seed{seed}")
    return True


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
    split_indices = {
        split: set(map(int, dataset.get_subset(split).indices))
        for split in ("train", "id_test", "val", "test")
    }
    overlaps = {}
    names = tuple(split_indices)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            count = len(split_indices[left] & split_indices[right])
            overlaps[f"{left}__{right}"] = count
            if count:
                raise ValueError(f"RxRx1 split-index leakage: {left}/{right} share {count}")
    subset = audit_environment_subset(dataset, quarter_environment_subset())
    if subset["n_classes_observed"] != subset["n_classes_expected"]:
        raise ValueError("quarter subset loses perturbation classes")
    if set(map(int, subset["cell_environment_counts"])) != {0, 1, 2, 3}:
        raise ValueError("quarter subset loses cell types")
    return {"quarter": subset, "split_index_counts": {k: len(v) for k, v in split_indices.items()},
            "split_index_overlaps": overlaps}


def write_manifest(out, rows, config=CONFIG):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    anchors = {(row[1], int(row[2])): row[4] for row in full_anchor_rows(config)}
    payload = {
        "schema_version": 1, "campaign": CAMPAIGN, "config": config,
        "expected_runs": 8, "seeds": list(SEEDS),
        "axis": "number_of_independent_training_experiments",
        "atomic_sampling_unit": "RxRx1 field, nested by experiment",
        "quarter_environment_subset": list(quarter_environment_subset()),
        "full_anchor_root": str(FULL_ANCHOR_ROOT),
        "checkpoint_rule": "terminal epoch 30; all predeclared arms receive stage-3 readout",
        "headline_endpoints": ["acc_heldout", "worst_env_heldout", "active_ffn_params"],
        "primary_contrast": "(shared-dense at full) - (shared-dense at quarter)",
        "stopping_rule": "exactly 30 epochs for every new arm; no adaptive extension",
        "dataset_audit": _dataset_audit(config),
        "runs": [
            {"label": label, "arm": arm, "seed": seed, "run_id": run_id,
             "environment_subset": list(environments), "overrides": overrides,
             "variant": cfg["model"]["variant"],
             "full_anchor_run_id": anchors[(arm, int(seed))]}
            for label, arm, seed, environments, overrides, run_id, cfg in rows
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
        raise RuntimeError("tracking unavailable (" + ", ".join(missing)
                           + "); pass --allow-untracked for local-first execution")
    if missing:
        env["WANDB_MODE"] = "offline"
        print(f"[tracking] local-first mode; missing: {', '.join(missing)}", flush=True)
    env["WANDB_GROUP"] = WANDB_GROUP
    env["WANDB_JOB_TYPE"] = "rxrx1_domain_scaling_replicate30"
    env["WANDB_TAGS"] = "rxrx1,cell-dino,domain-scaling,replication,matched-controls,stage3"
    env["CCAS_HF_PREFIX"] = HF_PREFIX
    return env


def command_for(row, config, out):
    return [sys.executable, "scripts/run_ccas.py", "--config", config,
            "--results-dir", str(out), "--override", *row[4]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--result-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--full-root", default=str(FULL_ANCHOR_ROOT))
    parser.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"))
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--allow-untracked", action="store_true")
    args = parser.parse_args()

    out = Path(args.result_root).expanduser().resolve()
    rows = wave_rows(args.config)
    write_manifest(out, rows, args.config)
    if args.status:
        print(render_report(out, args.full_root))
        return
    selected = sharded_rows(rows, args.shard_index, args.num_shards)
    pending = [row for row in selected if not (out / f"{row[5]}.json").exists()]
    print(render_report(out, args.full_root), flush=True)
    print(f"shard {args.shard_index}/{args.num_shards}: {len(selected)} planned, "
          f"{len(pending)} pending on GPUs {args.gpus}", flush=True)
    if args.dry_run:
        for row in selected:
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
            env = dict(base_env, CUDA_VISIBLE_DEVICES=gpu)
            log_handle = open(out / f"{row[5]}.log", "a")
            process = subprocess.Popen(command_for(row, args.config, out), env=env,
                                       stdout=log_handle, stderr=subprocess.STDOUT)
            gpulease.adopt(gpu, process.pid)
            running[gpu] = (process, row, log_handle)
            print(f"[start] gpu={gpu} pid={process.pid} {row[0]} {row[5]}", flush=True)
        for gpu in list(running):
            process, row, log_handle = running[gpu]
            if process.poll() is not None:
                log_handle.close()
                gpulease.release(gpu, pid=process.pid)
                print(f"[exit] gpu={gpu} rc={process.returncode} {row[0]} {row[5]}", flush=True)
                print(render_report(out, args.full_root), flush=True)
                del running[gpu]
        if pending or running:
            time.sleep(10)


if __name__ == "__main__":
    main()
