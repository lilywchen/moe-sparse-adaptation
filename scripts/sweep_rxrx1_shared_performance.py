#!/usr/bin/env python
"""Eight-run Cell-DINO shared/residual-MoE performance wave for four 2xH100 containers.

Six arms stay within standard MoE design axes: replacement versus shared/residual allocation,
top-k, expert count, and depth.  Two bounded arms add established robustness tools separately:
supervised cross-experiment contrastive consistency and MixStyle feature-stat augmentation.
Every terminal run evaluates both OOD validation and OOD test, uploads its validated artifact
folder to Hugging Face, and logs live to W&B when the credentials already configured on SciServer
are available.
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

from moe_shift.capacity.naming import run_id_from
from moe_shift.utils import gpulease
from moe_shift.utils.config import apply_overrides, load_config
from summarize_rxrx1_performance_wave import render_table


CONFIG = "configs/ccas_rxrx1_cell_dino_native.yaml"
CAMPAIGN = "shared_residual_performance30_20260807"
DEFAULT_ROOT = Path(
    "/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/"
    "substrate_rxrx1/cell_dino_cp5/shared_residual_performance30_20260807"
)
WANDB_GROUP = "rxrx1-cell-dino-shared-residual-performance30-20260807"
HF_PREFIX = "rxrx1/cell_dino_cp5/shared_residual_performance30_20260807"


def _blocks(indices):
    return "model.ffn_block_indices=[" + ",".join(str(index) for index in indices) + "]"


def _common(label):
    return [
        "seed=0", "stage=3", "model.routing_estimator=selected_st",
        "model.routing_unit=token", "model.geometry=cosine",
        "model.pressure=canonical", "model.balance=global",
        "model.freeze_backbone=false", "model.unfreeze_last_n_blocks=0",
        "model.sym_break_moe=0.0", "model.feature_stat_mix_prob=0.0",
        "train.objective=erm", "train.cross_experiment_pairs=false",
        "train.epochs=30", "train.milestone_epochs=[5,10,20,30]",
        "train.save_checkpoint_epochs=[30]", "train.warmup_epochs=3",
        "train.llrd=1.0", "train.batch_size=64", "train.optim.lr=1.0e-4",
        "train.optim.weight_decay=0.05", "model.drop_path=0.1",
        "losses.balance_w=1.0e-2", "losses.zloss_w=1.0e-3",
        "losses.cross_experiment_contrastive_w=0.0",
        "losses.cross_experiment_contrastive_temperature=0.1",
        "analysis.run_mechanism=false", "analysis.record_train_accuracy=true",
        f"run_tag={CAMPAIGN}_{label}",
    ]


def wave_rows(config=CONFIG):
    specs = [
        ("replace_E4k2_late2", [
            "model.variant=moe", "model.n_experts=4", "model.top_k=2", _blocks((10, 11)),
        ]),
        ("shared_E3k1_late2", [
            "model.variant=shared_moe", "model.n_experts=3", "model.top_k=1", _blocks((10, 11)),
        ]),
        ("shared_E3k2_late2", [
            "model.variant=shared_moe", "model.n_experts=3", "model.top_k=2", _blocks((10, 11)),
        ]),
        ("shared_E7k1_late2", [
            "model.variant=shared_moe", "model.n_experts=7", "model.top_k=1", _blocks((10, 11)),
        ]),
        ("shared_E3k1_late4", [
            "model.variant=shared_moe", "model.n_experts=3", "model.top_k=1",
            _blocks((8, 9, 10, 11)),
        ]),
        ("shared_E3k2_late4", [
            "model.variant=shared_moe", "model.n_experts=3", "model.top_k=2",
            _blocks((8, 9, 10, 11)),
        ]),
        ("shared_E3k1_xbatch", [
            "model.variant=shared_moe", "model.n_experts=3", "model.top_k=1", _blocks((10, 11)),
            "train.cross_experiment_pairs=true",
            "losses.cross_experiment_contrastive_w=0.1",
        ]),
        ("shared_E3k1_mixstyle", [
            "model.variant=shared_moe", "model.n_experts=3", "model.top_k=1", _blocks((10, 11)),
            "model.feature_stat_mix_prob=0.5", "model.feature_stat_mix_alpha=0.1",
        ]),
    ]
    rows = []
    for label, intervention in specs:
        overrides = [*_common(label), *intervention]
        cfg = apply_overrides(load_config(config), overrides)
        rows.append((label, overrides, run_id_from(cfg), cfg))
    return rows


def sharded_rows(rows, shard_index, num_shards):
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("require 0 <= shard_index < num_shards")
    return [row for index, row in enumerate(rows) if index % num_shards == shard_index]


def write_manifest(out, rows):
    out.mkdir(parents=True, exist_ok=True)
    path = out / "wave_manifest.json"
    payload = {
        "schema_version": 1, "campaign": CAMPAIGN, "config": CONFIG,
        "selection_split": "ood_val", "test_readout": "all_predefined_arms",
        "runs": [
            {"label": label, "run_id": run_id, "overrides": overrides}
            for label, overrides, run_id, _cfg in rows
        ],
    }
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    os.replace(temporary, path)


def tracking_environment():
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
    if missing:
        raise RuntimeError(
            "tracking is required for this wave; missing configured value(s): "
            + ", ".join(missing))
    env["WANDB_GROUP"] = WANDB_GROUP
    env["WANDB_JOB_TYPE"] = "rxrx1_shared_residual_performance30"
    env["WANDB_TAGS"] = "rxrx1,cell-dino,shared-expert,residual-moe,performance,stage3"
    env["CCAS_HF_PREFIX"] = HF_PREFIX
    return env


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
    args = parser.parse_args()

    out = Path(args.result_root).expanduser().resolve()
    all_rows = wave_rows(args.config)
    write_manifest(out, all_rows)
    if args.status:
        print(render_table(out))
        return
    rows = sharded_rows(all_rows, args.shard_index, args.num_shards)
    pending = [row for row in rows if not (out / f"{row[2]}.json").exists()]
    print(render_table(out), flush=True)
    print(
        f"shard {args.shard_index}/{args.num_shards}: {len(rows)} planned, "
        f"{len(pending)} pending on GPUs {args.gpus}", flush=True)
    if args.dry_run:
        for label, _, run_id, _cfg in rows:
            print(f"  {label}: {run_id}")
        return

    base_env = tracking_environment()
    slots = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    running = {}
    while pending or running:
        while pending and len(running) < args.max_concurrent:
            gpu = gpulease.acquire_any(slots)
            if gpu is None:
                break
            label, overrides, run_id, _cfg = pending.pop(0)
            env = dict(base_env, CUDA_VISIBLE_DEVICES=gpu)
            log_handle = open(out / f"{run_id}.log", "a")
            command = [
                sys.executable, "scripts/run_ccas.py", "--config", args.config,
                "--results-dir", str(out), "--override", *overrides,
            ]
            process = subprocess.Popen(
                command, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
            gpulease.adopt(gpu, process.pid)
            running[gpu] = (process, run_id, label, log_handle)
            print(f"[start] gpu={gpu} pid={process.pid} {label} {run_id}", flush=True)

        for gpu in list(running):
            process, run_id, label, log_handle = running[gpu]
            if process.poll() is not None:
                log_handle.close()
                gpulease.release(gpu, pid=process.pid)
                print(f"[exit] gpu={gpu} rc={process.returncode} {label} {run_id}", flush=True)
                print(render_table(out), flush=True)
                del running[gpu]
        if pending or running:
            time.sleep(10)


if __name__ == "__main__":
    main()
