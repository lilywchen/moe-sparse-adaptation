#!/usr/bin/env python3
"""Predeclared E1/E3-router/dense-E2 conditionality gate on Cell-DINO/RxRx1."""
from __future__ import annotations
import argparse, gc, json, os, subprocess, sys, time, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from aggregate_rxrx1_conditionality_gate import render_report
from moe_shift.capacity.naming import run_id_from
from moe_shift.data.rxrx1_scaling import audit_environment_subset, full_environment_subset
from moe_shift.utils import gpulease
from moe_shift.utils.config import apply_overrides, load_config

CONFIG = "configs/ccas_rxrx1_cell_dino_native.yaml"
CAMPAIGN = "rxrx1_conditionality_gate30_20260810"
DEFAULT_ROOT = Path("/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/"
                    "substrate_rxrx1/cell_dino_cp5/rxrx1_conditionality_gate30_20260810")
WANDB_GROUP = "rxrx1-cell-dino-conditionality-gate30-20260810"
HF_PREFIX = "rxrx1/cell_dino_cp5/rxrx1_conditionality_gate30_20260810"
SEEDS, LATE2 = (11, 23, 37), (10, 11)
ARMS = (
    ("shared_E1_unconditional", ["model.variant=shared_moe", "model.n_experts=1",
                                  "model.top_k=1", "model.routing_estimator=selected_st",
                                  "model.ffn_block_indices=[10,11]"]),
    ("shared_E3_selected", ["model.variant=shared_moe", "model.n_experts=3",
                            "model.top_k=1", "model.routing_estimator=selected_st",
                            "model.ffn_block_indices=[10,11]"]),
    ("shared_E3_fullST", ["model.variant=shared_moe", "model.n_experts=3",
                          "model.top_k=1", "model.routing_estimator=full_st",
                          "model.ffn_block_indices=[10,11]"]),
    ("dense_E2_active_matched", ["model.variant=dense_wide", "model.n_experts=2",
                                 "model.top_k=1", "model.routing_estimator=selected_st",
                                 "model.ffn_block_indices=[10,11]"]),
)
ALLOWED_ARCHITECTURE_DIFFERENCES = {
    "model.variant", "model.n_experts", "model.routing_estimator", "run_tag"}

def _source_identity():
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip())
    return sha, dirty

def _common(seed, label):
    return [f"seed={seed}", f"train.model_seed={seed}", f"train.data_seed={seed}",
            f"train.training_seed={seed}", "stage=3", "model.routing_unit=token",
            "model.geometry=cosine", "model.pressure=canonical", "model.balance=global",
            "model.freeze_backbone=false", "model.unfreeze_last_n_blocks=0",
            "model.sym_break_moe=0.0", "model.feature_stat_mix_prob=0.0",
            "model.router_frozen=false", "train.objective=erm",
            "train.cross_experiment_pairs=false", "train.epochs=30",
            "train.milestone_epochs=[5,10,20,30]", "train.save_checkpoint_epochs=[30]",
            "train.warmup_epochs=3", "train.llrd=1.0", "train.batch_size=64",
            "train.num_workers=8", "train.optim.lr=1.0e-4",
            "train.optim.weight_decay=0.05", "train.label_smoothing=0.1",
            "model.drop_path=0.1", "losses.balance_w=1.0e-2", "losses.zloss_w=1.0e-3",
            "losses.cross_experiment_contrastive_w=0.0", "analysis.run_mechanism=true",
            "analysis.record_train_accuracy=true", f"run_tag={CAMPAIGN}_{label}"]

def wave_rows(config=CONFIG):
    rows = []
    for seed in SEEDS:
        for arm, intervention in ARMS:
            label = f"{arm}_s{seed}"; overrides = [*_common(seed, label), *intervention]
            cfg = apply_overrides(load_config(config), overrides)
            rows.append((label, arm, seed, overrides, run_id_from(cfg), cfg))
    if len(rows) != 12 or len({r[4] for r in rows}) != 12:
        raise ValueError("conditionality gate requires twelve collision-free run ids")
    validate_resolved_configs(rows); return rows

def _flatten(value, prefix=""):
    output = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key); output.update(_flatten(child, path))
    else: output[prefix] = value
    return output

def validate_resolved_configs(rows):
    for seed in SEEDS:
        same_seed = [r for r in rows if r[2] == seed]; reference = _flatten(same_seed[0][5])
        for row in same_seed[1:]:
            current = _flatten(row[5]); differences = {k for k in set(reference)|set(current)
                                                       if reference.get(k) != current.get(k)}
            unexpected = differences - ALLOWED_ARCHITECTURE_DIFFERENCES
            if unexpected: raise ValueError(f"unexpected config drift for {row[0]}: {sorted(unexpected)}")
    return True

def sharded_rows(rows, shard_index, num_shards):
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("require 0 <= shard_index < num_shards")
    return [row for index, row in enumerate(rows) if index % num_shards == shard_index]

def dataset_audit(config=CONFIG):
    from wilds import get_dataset
    cfg = load_config(config)
    dataset = get_dataset(dataset="rxrx1", root_dir=cfg["data_root"], download=False)
    audit = audit_environment_subset(dataset, full_environment_subset())
    if audit["n_classes_observed"] != int(cfg["model"]["num_classes"]):
        raise ValueError("full RxRx1 split does not preserve 1,139 labels")
    if set(audit["cell_environment_counts"]) != {0, 1, 2, 3}:
        raise ValueError("full RxRx1 split does not preserve all cell types")
    return {"passed": True, "atomic_unit": "WILDS field/site record", **audit}

def capacity_accounting(rows):
    from moe_shift.capacity.model import build_ccas
    reports = {}
    for arm, _ in ARMS:
        model = build_ccas(next(r[5] for r in rows if r[1] == arm)); report = model.capacity.as_dict()
        report["inference_active_ffn_params"] = report["active_ffn_params"]
        report["training_active_converted_ffn_params"] = (
            report["ffn_block_params"] if arm == "shared_E3_fullST" else report["active_ffn_params"])
        reports[arm] = report; del model; gc.collect()
    active = [reports[a]["inference_active_ffn_params"] for a in
              ("shared_E1_unconditional", "shared_E3_selected", "dense_E2_active_matched")]
    if max(active)/min(active) > 1.001: raise ValueError("inference-active budgets differ by >0.1%")
    return reports

def write_manifest(out, rows, config=CONFIG, audit=None, capacity=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True); sha, dirty = _source_identity()
    if dirty: raise ValueError("conditionality manifest requires a clean tracked worktree")
    payload = {"schema_version": 1, "campaign": CAMPAIGN, "config": config,
      "expected_runs": 12, "seeds": list(SEEDS),
      "scientific_question": "Does learned conditional banking beat an unconditional residual or active-inference-compute-matched dense adapter, and was the historical surrogate the bottleneck?",
      "primary_contrasts": ["fullST-selected", "fullST-E1", "selected-E1", "fullST-denseE2"],
      "headline_endpoints": ["OOD test accuracy", "worst-decile OOD experiment accuracy",
                             "paired per-experiment deltas", "inference-active FFN parameters"],
      "selection_and_test_rule": "fixed 30 epochs; all arms predeclared; test read once; no adaptive topology",
      "stopping_rule": "Only generic-router salvage wave: stop if E3 does not beat E1 and fullST does not beat dense E2 with positive paired uncertainty.",
      "label_smoothing_fixed": .1,
      "rng_pairing": "explicit equal model/data/training seeds within each architecture quartet",
      "training_compute_caveat": "fullST is dense during training and top-1 sparse at evaluation; both budgets reported separately",
      "allowed_config_differences": sorted(ALLOWED_ARCHITECTURE_DIFFERENCES | {"seed"}),
      "source_git_commit": sha, "source_git_dirty": dirty,
      "dataset_audit": dataset_audit(config) if audit is None else audit,
      "compute_accounting": capacity_accounting(rows) if capacity is None else capacity,
      "runs": [{"label": l, "arm": a, "seed": s, "overrides": o, "run_id": rid,
                "variant": c["model"]["variant"], "resolved_config": c}
               for l, a, s, o, rid, c in rows]}
    path = out/"wave_manifest.json"; tmp = path.with_name(path.name+f".{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True)); os.replace(tmp, path); return payload

def tracking_environment(require_tracking=True):
    env = dict(os.environ)
    if not env.get("HF_TOKEN"):
        try:
            from huggingface_hub import get_token; env["HF_TOKEN"] = get_token() or ""
        except Exception: pass
    if not env.get("WANDB_API_KEY"):
        try:
            import wandb; env["WANDB_API_KEY"] = wandb.api.api_key or ""
        except Exception: pass
    missing = [n for n in ("WANDB_API_KEY", "HF_TOKEN", "CCAS_HF_REPO") if not env.get(n)]
    if missing and require_tracking: raise RuntimeError("tracking unavailable: "+", ".join(missing))
    if missing: env["WANDB_MODE"] = "offline"; print("[tracking] local-first", flush=True)
    env.update(WANDB_GROUP=WANDB_GROUP, WANDB_JOB_TYPE="rxrx1_conditionality_gate30",
               WANDB_TAGS="rxrx1,cell-dino,conditionality,full-st,active-compute,stage3",
               CCAS_HF_PREFIX=HF_PREFIX); return env

def command_for(row, config, out):
    return [sys.executable, "scripts/run_ccas.py", "--config", config,
            "--results-dir", str(out), "--override", *row[3]]

def main():
    p = argparse.ArgumentParser(); p.add_argument("--config", default=CONFIG)
    p.add_argument("--result-root", default=str(DEFAULT_ROOT)); p.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"))
    p.add_argument("--max-concurrent", type=int, default=2); p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1); p.add_argument("--dry-run", action="store_true")
    p.add_argument("--status", action="store_true"); p.add_argument("--allow-untracked", action="store_true")
    args = p.parse_args(); out = Path(args.result_root).expanduser().resolve()
    if args.status: print(render_report(out)); return
    all_rows = wave_rows(args.config); write_manifest(out, all_rows, args.config)
    rows = sharded_rows(all_rows, args.shard_index, args.num_shards)
    pending = [r for r in rows if not (out/f"{r[4]}.json").exists()]
    print(render_report(out), flush=True); print(f"shard {args.shard_index}/{args.num_shards}: {len(rows)} planned, {len(pending)} pending", flush=True)
    if args.dry_run:
        for r in rows: print(f"  {r[0]}: {r[4]}")
        return
    slots = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if args.max_concurrent > len(slots): raise ValueError("max-concurrent exceeds GPU slots")
    base_env, running = tracking_environment(not args.allow_untracked), {}
    while pending or running:
        while pending and len(running) < args.max_concurrent:
            gpu = gpulease.acquire_any(slots)
            if gpu is None: break
            row = pending.pop(0); label, rid = row[0], row[4]; env = dict(base_env, CUDA_VISIBLE_DEVICES=gpu)
            log = open(out/f"{rid}.log", "a"); proc = subprocess.Popen(command_for(row, args.config, out), env=env, stdout=log, stderr=subprocess.STDOUT)
            gpulease.adopt(gpu, proc.pid); running[gpu] = (proc, rid, label, log)
            print(f"[start] gpu={gpu} pid={proc.pid} {label} {rid}", flush=True)
        for gpu in list(running):
            proc, rid, label, log = running[gpu]
            if proc.poll() is not None:
                log.close(); gpulease.release(gpu, pid=proc.pid)
                print(f"[exit] gpu={gpu} rc={proc.returncode} {label} {rid}", flush=True)
                print(render_report(out), flush=True); del running[gpu]
        if pending or running: time.sleep(10)

if __name__ == "__main__": main()
