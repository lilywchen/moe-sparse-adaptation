#!/usr/bin/env python3
"""Matched 2-seed competence pilot on the full RxRx3-core plate/guide point."""

import argparse
import gc
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

from aggregate_rxrx3_core_pilot import render_report
from moe_shift.capacity.naming import run_id_from
from moe_shift.data.rxrx3_core import read_rxrx3_manifest
from moe_shift.utils import gpulease
from moe_shift.utils.config import apply_overrides, load_config


CONFIG = "configs/ccas_rxrx3_core_cell_dino.yaml"
CAMPAIGN = "rxrx3_core_pilot10_20260810"
DEFAULT_ROOT = Path(
    "/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/"
    "substrate_rxrx3_core/cell_dino_cp5/rxrx3_core_pilot10_20260810"
)
WANDB_GROUP = "rxrx3-core-cell-dino-pilot10-20260810"
HF_PREFIX = "rxrx3_core/cell_dino_cp5/pilot10_20260810"
SEEDS = (1, 2)
ARMS = (
    ("original", ["model.variant=original", "model.ffn_block_indices=[10,11]"]),
    ("dense_E4_late2", ["model.variant=dense_wide", "model.n_experts=4",
                         "model.ffn_block_indices=[10,11]"]),
    ("replace_E4k2_late2", ["model.variant=moe", "model.n_experts=4", "model.top_k=2",
                             "model.ffn_block_indices=[10,11]"]),
    ("shared_E3k1_late2", ["model.variant=shared_moe", "model.n_experts=3", "model.top_k=1",
                            "model.ffn_block_indices=[10,11]"]),
)
ALLOWED_ARCHITECTURE_DIFFERENCES = {
    "model.variant", "model.n_experts", "model.top_k", "model.ffn_block_indices", "run_tag",
}


def _source_identity():
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip())
    return sha, dirty


def _common(seed, label):
    return [
        f"seed={seed}", "stage=3", "model.routing_estimator=selected_st",
        "model.routing_unit=token", "model.geometry=cosine", "model.pressure=canonical",
        "model.balance=global", "model.freeze_backbone=false",
        "model.unfreeze_last_n_blocks=0", "model.sym_break_moe=0.0",
        "model.feature_stat_mix_prob=0.0", "model.router_frozen=false",
        "train.objective=erm", "train.cross_experiment_pairs=false", "train.epochs=10",
        "train.milestone_epochs=[5,10]", "train.save_checkpoint_epochs=[10]",
        "train.warmup_epochs=1", "train.llrd=1.0", "train.batch_size=64",
        "train.num_workers=8", "train.optim.lr=1.0e-4", "train.optim.weight_decay=0.05",
        "train.label_smoothing=0.0", "model.drop_path=0.1",
        "losses.balance_w=1.0e-2", "losses.zloss_w=1.0e-3",
        "losses.cross_experiment_contrastive_w=0.0", "analysis.run_mechanism=false",
        "analysis.record_train_accuracy=true", f"run_tag={CAMPAIGN}_{label}",
    ]


def wave_rows(config=CONFIG):
    rows = []
    for seed in SEEDS:
        for arm, intervention in ARMS:
            label = f"{arm}_s{seed}"
            overrides = [*_common(seed, label), *intervention]
            cfg = apply_overrides(load_config(config), overrides)
            rows.append((label, arm, seed, overrides, run_id_from(cfg), cfg))
    if len(rows) != 8 or len({row[4] for row in rows}) != 8:
        raise ValueError("RxRx3 pilot must contain exactly eight collision-free run ids")
    validate_resolved_configs(rows)
    return rows


def _flatten(value, prefix=""):
    output = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            output.update(_flatten(child, path))
    else:
        output[prefix] = value
    return output


def validate_resolved_configs(rows):
    """Fail if same-seed arms drift outside the predeclared architecture fields."""
    for seed in SEEDS:
        same_seed = [row for row in rows if row[2] == seed]
        reference = _flatten(same_seed[0][5])
        for row in same_seed[1:]:
            current = _flatten(row[5])
            differences = {
                key for key in set(reference) | set(current)
                if reference.get(key) != current.get(key)
            }
            unexpected = differences - ALLOWED_ARCHITECTURE_DIFFERENCES
            if unexpected:
                raise ValueError(f"unexpected config drift for {row[0]}: {sorted(unexpected)}")
    return True


def sharded_rows(rows, shard_index, num_shards):
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("require 0 <= shard_index < num_shards")
    return [row for index, row in enumerate(rows) if index % num_shards == shard_index]


def dataset_audit(config=CONFIG):
    cfg = load_config(config)
    records, summary = read_rxrx3_manifest(cfg["rxrx3_manifest"])
    audit_path = Path(cfg["rxrx3_data_dir"]).parent / "rxrx3_core_image_audit.json"
    audit = json.loads(audit_path.read_text())
    checks = {
        "image_audit": audit.get("passed") is True,
        "classes": summary["classes"] == 674,
        "train_rows": summary["split_counts"]["train"] == 21404,
        "id_val_rows": summary["split_counts"]["id_val"] == 2708,
        "ood_test_rows": summary["split_counts"]["ood_test"] == 23855,
        "train_experiments": summary["train_experiments"] == 85,
        "ood_test_experiments": summary["ood_test_experiments"] == 85,
        "manifest_unique_wells": len(records) == len({row["well_id"] for row in records}),
        "six_channel_audit_rows": audit.get("channel_rows") == 1335606,
        "six_channel_manifest_union": audit.get("manifest_union_wells") == 47967,
    }
    if not all(checks.values()):
        raise ValueError(f"RxRx3 dataset gate failed: {checks}")
    return {"passed": True, "checks": checks, **{
        key: summary[key] for key in (
            "manifest", "manifest_sha256", "well_set_sha256", "split_counts", "classes",
            "train_experiments", "ood_test_experiments", "cell_types",
        )
    }}


def capacity_accounting(rows):
    from moe_shift.capacity.model import build_ccas
    reports = {}
    for arm in (name for name, _ in ARMS):
        cfg = next(row[5] for row in rows if row[1] == arm)
        model = build_ccas(cfg)
        reports[arm] = model.capacity.as_dict()
        del model
        gc.collect()
    reference = reports["shared_E3k1_late2"]["active_ffn_params"]
    for report in reports.values():
        report["estimated_active_ffn_flops_relative"] = (
            report["active_ffn_params"] / reference
        )
    return reports


def write_manifest(out, rows, config=CONFIG, audit=None, capacity=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    source_sha, source_dirty = _source_identity()
    if source_dirty:
        raise ValueError("RxRx3 pilot manifest requires a clean tracked worktree")
    audit = dataset_audit(config) if audit is None else audit
    capacity = capacity_accounting(rows) if capacity is None else capacity
    payload = {
        "schema_version": 1, "campaign": CAMPAIGN, "config": config,
        "expected_runs": 8, "seeds": list(SEEDS),
        "task": "674-way CRISPR gene perturbation identification",
        "atomic_unit": "well with exactly six joined stain rows",
        "training_point": "8 plates and 4 guides per gene (full frozen manifest)",
        "selection_split": "fixed ID-validation plate in each train experiment",
        "headline_endpoints": ["acc_heldout", "worst_env_heldout", "active_ffn_params"],
        "primary_contrast": "shared-residual E3/top-1 versus dense E4 at matched total capacity",
        "competence_gate": "each arm train >=5% and ID validation >=1%; no NaN/fatal artifact",
        "stopping_rule": "exactly 10 epochs for all eight arms; no adaptive topology addition",
        "allowed_config_differences": sorted(ALLOWED_ARCHITECTURE_DIFFERENCES | {"seed"}),
        "source_git_commit": source_sha, "source_git_dirty": source_dirty,
        "dataset_audit": audit, "compute_accounting": capacity,
        "runs": [
            {"label": label, "arm": arm, "seed": seed, "overrides": overrides,
             "run_id": run_id, "variant": cfg["model"]["variant"], "resolved_config": cfg}
            for label, arm, seed, overrides, run_id, cfg in rows
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
    missing = [name for name in ("WANDB_API_KEY", "HF_TOKEN", "CCAS_HF_REPO") if not env.get(name)]
    if missing and require_tracking:
        raise RuntimeError("tracking unavailable (" + ", ".join(missing) +
                           "); pass --allow-untracked for local-first execution")
    if missing:
        env["WANDB_MODE"] = "offline"
        print(f"[tracking] local-first mode; missing: {', '.join(missing)}", flush=True)
    env["WANDB_GROUP"] = WANDB_GROUP
    env["WANDB_JOB_TYPE"] = "rxrx3_core_pilot10"
    env["WANDB_TAGS"] = "rxrx3-core,cell-dino,pilot,matched-controls,stage3"
    env["CCAS_HF_PREFIX"] = HF_PREFIX
    return env


def command_for(row, config, out):
    return [sys.executable, "scripts/run_ccas.py", "--config", config,
            "--results-dir", str(out), "--override", *row[3]]


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
    rows = wave_rows(args.config)
    write_manifest(out, rows, args.config)
    if args.status:
        print(render_report(out))
        return
    selected = sharded_rows(rows, args.shard_index, args.num_shards)
    pending = [row for row in selected if not (out / f"{row[4]}.json").exists()]
    print(render_report(out), flush=True)
    print(f"shard {args.shard_index}/{args.num_shards}: {len(selected)} planned, "
          f"{len(pending)} pending on GPUs {args.gpus}", flush=True)
    if args.dry_run:
        for row in selected:
            print(f"  {row[0]}: {row[4]}")
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
            log_handle = open(out / f"{row[4]}.log", "a")
            process = subprocess.Popen(command_for(row, args.config, out), env=env,
                                       stdout=log_handle, stderr=subprocess.STDOUT)
            gpulease.adopt(gpu, process.pid)
            running[gpu] = (process, row, log_handle)
            print(f"[start] gpu={gpu} pid={process.pid} {row[0]} {row[4]}", flush=True)
        for gpu in list(running):
            process, row, log_handle = running[gpu]
            if process.poll() is not None:
                log_handle.close()
                gpulease.release(gpu, pid=process.pid)
                print(f"[exit] gpu={gpu} rc={process.returncode} {row[0]} {row[4]}", flush=True)
                print(render_report(out), flush=True)
                del running[gpu]
        if pending or running:
            time.sleep(10)


if __name__ == "__main__":
    main()

