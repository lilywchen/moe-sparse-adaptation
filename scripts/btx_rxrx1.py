#!/usr/bin/env python
"""Branch-Train-MiX for RxRx1: make MoE experts distinct by CONSTRUCTION.

Every completed MoE arm initialises its experts from one shared FFN (``copy.deepcopy(mlp)`` with
``sym_break_moe=0.0``) and relies on gradient descent to separate them.  The measurements say that
does not happen -- ``route_reliance <= 0.0065`` against a ``0.01`` gate, learned routing matched
frozen routing, and the canonical balance loss is satisfied perfectly by N identical experts used
uniformly, so nothing in the objective ever pushed them apart.

BTX (Sukhbaatar et al., 2024) removes the assumption in three phases:

``cluster``
    Partition the 33 training experiments.  ``feature_mean`` uses each environment's mean backbone
    embedding (cheap, no gradients); ``gradient_conflict`` consumes an existing pairwise cosine
    matrix so the campaign's own conflict measurement decides which experiments share an expert;
    ``file`` takes a frozen hand partition.

``specialists``
    Fine-tune one INDEPENDENT specialist per cluster, on that cluster's environments only.  Each
    is an ordinary ``variant=original`` run, so it inherits the existing milestone, validation and
    provenance machinery rather than a parallel training path.

``mix``
    Load the specialists' FFN weights into a ``shared_moe`` expert bank, freeze them, and train
    only the router.  Experts are now guaranteed distinct because they were fitted to disjoint
    data, so this phase is a clean test of whether a router can exploit genuinely different
    experts -- the question every previous arm confounded with "do experts differentiate at all".

Deliberately NOT function-preserving at initialisation of the mix phase: the routed residual
carries the specialists' learned deviation, which is the entire point.  That is recorded in the
manifest rather than silently assumed.
"""
import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe_shift.capacity.btx import (
    MANIFEST_NAME,
    cluster_environments,
    cluster_from_conflict_matrix,
    load_clusters,
    write_manifest,
)
from moe_shift.utils.config import apply_overrides, load_config

CLUSTERS_NAME = "btx_clusters.json"


# ------------------------------------------------------------------------------- phase 1
def compute_environment_descriptors(cfg, max_batches_per_env=4):
    """Mean pretrained-backbone embedding per TRAINING environment.

    Training data only, no gradients, and no validation or test access -- the partition is a
    property of the training set, so letting held-out data influence it would contaminate every
    downstream comparison.
    """
    import torch

    from moe_shift.capacity.model import build_ccas
    from moe_shift.data import make_loaders

    probe_cfg = json.loads(json.dumps(cfg))          # deep copy; build_ccas mutates nothing else
    probe_cfg["model"]["variant"] = "original"
    train_loader, _within, _heldout, _audit = make_loaders(probe_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_ccas(probe_cfg).to(device)
    model.eval()

    sums = defaultdict(lambda: None)
    counts = defaultdict(int)
    seen_batches = defaultdict(int)
    with torch.no_grad():
        for batch in train_loader:
            x = batch[0].to(device)
            env = batch[3]
            feats = model.forward_features(x).float().cpu()
            for raw in torch.unique(env):
                key = int(raw)
                if seen_batches[key] >= max_batches_per_env:
                    continue
                mask = (env == raw)
                block = feats[mask].sum(dim=0)
                sums[key] = block if sums[key] is None else sums[key] + block
                counts[key] += int(mask.sum())
                seen_batches[key] += 1
            if counts and all(seen_batches[k] >= max_batches_per_env for k in seen_batches) \
                    and len(seen_batches) >= int(cfg["sites"]["K"]):
                break
    if not sums:
        raise RuntimeError("no training environments observed while probing descriptors")
    return {key: (value / max(counts[key], 1)).tolist() for key, value in sums.items()}


def phase_cluster(args, cfg):
    out = Path(args.results_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    path = out / CLUSTERS_NAME
    if path.is_file() and not args.force:
        print(f"[btx] reusing existing clusters {path}")
        return load_clusters(path)

    if args.cluster_source == "file":
        if not args.clusters_json:
            raise ValueError("--cluster-source file requires --clusters-json")
        clusters = load_clusters(args.clusters_json)
    elif args.cluster_source == "gradient_conflict":
        if not args.conflict_json:
            raise ValueError("--cluster-source gradient_conflict requires --conflict-json")
        payload = json.loads(Path(args.conflict_json).read_text())
        matrix = payload.get("pairwise_environment_cosine") or payload.get("matrix")
        if not matrix:
            raise ValueError(
                f"{args.conflict_json} has no 'pairwise_environment_cosine' matrix; the "
                "gradient-conflict profile must be exported with per-environment cosines")
        clusters = cluster_from_conflict_matrix(
            matrix, n_clusters=args.n_clusters, seed=int(cfg["seed"]))
    else:
        descriptors = compute_environment_descriptors(cfg)
        clusters = cluster_environments(
            descriptors, n_clusters=args.n_clusters, seed=int(cfg["seed"]))

    payload = {
        "schema_version": 1,
        "cluster_source": args.cluster_source,
        "n_clusters": len(clusters),
        "seed": int(cfg["seed"]),
        "data_scope": "train_only",
        "test_evaluated": False,
        "clusters": {str(k): v for k, v in clusters.items()},
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"[btx] wrote {len(clusters)} clusters -> {path}")
    for index, members in sorted(clusters.items()):
        print(f"       cluster {index}: {len(members)} environments {members}")
    return clusters


# ------------------------------------------------------------------------------- phase 2
def phase_specialists(args, clusters):
    """Train one independent specialist per cluster via the ordinary runner."""
    out = Path(args.results_dir).expanduser().resolve()
    specialists = []
    for index, members in sorted(clusters.items()):
        tag = f"{args.run_tag_prefix}_specialist{index}"
        overrides = [
            "model.variant=original",
            f"train.epochs={args.specialist_epochs}",
            f"train.milestone_epochs=[{args.specialist_epochs}]",
            f"train.save_checkpoint_epochs=[{args.specialist_epochs}]",
            "train.warmup_epochs=1",
            "stage=1",                                   # specialists never touch OOD test
            "analysis.run_mechanism=false",
            "analysis.record_train_accuracy=false",
            f"train.environment_subset={json.dumps(members)}",
            f"run_tag={tag}",
        ]
        cfg = apply_overrides(load_config(args.config), [*args.override, *overrides])
        from moe_shift.capacity.naming import run_id_from
        run_id = run_id_from(cfg)
        checkpoint = out / f"{run_id}.epoch{int(args.specialist_epochs):03d}.pt"
        if checkpoint.is_file() and not args.force:
            print(f"[btx] specialist {index} already trained: {checkpoint.name}")
        else:
            command = [sys.executable, "scripts/run_ccas.py", "--config", args.config,
                       "--results-dir", str(out), "--override", *args.override, *overrides]
            print(f"[btx] training specialist {index} on {len(members)} environments", flush=True)
            completed = subprocess.run(command, cwd=str(ROOT))
            if completed.returncode != 0:
                raise RuntimeError(f"specialist {index} failed with rc={completed.returncode}")
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"specialist {index} produced no checkpoint at {checkpoint}")
        specialists.append({
            "cluster": index, "run_id": run_id, "environments": members,
            "checkpoint": str(checkpoint), "epochs": int(args.specialist_epochs),
        })
    manifest_path = out / MANIFEST_NAME
    write_manifest(manifest_path, clusters, specialists, args.cluster_source,
                   extra={"specialist_epochs": int(args.specialist_epochs),
                          "data_scope": "train_only", "test_evaluated": False})
    print(f"[btx] wrote manifest -> {manifest_path}")
    return manifest_path


# ------------------------------------------------------------------------------- phase 3
def phase_mix(args, manifest_path, n_experts):
    """Train the router over the frozen specialist bank, then optionally unfreeze."""
    out = Path(args.results_dir).expanduser().resolve()
    overrides = [
        *args.override,
        "model.variant=shared_moe",
        f"model.n_experts={n_experts}",
        "model.top_k=1",
        f"model.btx_manifest={manifest_path}",
        f"model.btx_freeze_experts={'true' if args.freeze_experts else 'false'}",
    ]
    command = [sys.executable, "scripts/run_ccas.py", "--config", args.config,
               "--results-dir", str(out), "--override", *overrides]
    print(f"[btx] mixing {n_experts} specialists; router-only={args.freeze_experts}", flush=True)
    completed = subprocess.run(command, cwd=str(ROOT))
    if completed.returncode != 0:
        raise RuntimeError(f"BTX mix phase failed with rc={completed.returncode}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("cluster", "specialists", "mix", "run-all"))
    parser.add_argument("--config", default="configs/ccas_rxrx1_cell_dino_native.yaml")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--n-clusters", type=int, default=4)
    parser.add_argument("--cluster-source", default="feature_mean",
                        choices=("feature_mean", "gradient_conflict", "file"))
    parser.add_argument("--clusters-json", default=None)
    parser.add_argument("--conflict-json", default=None)
    parser.add_argument("--specialist-epochs", type=int, default=5)
    parser.add_argument("--freeze-experts", action="store_true", default=True)
    parser.add_argument("--unfreeze-experts", dest="freeze_experts", action="store_false")
    parser.add_argument("--run-tag-prefix", default="btx")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    out = Path(args.results_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.phase == "cluster":
        phase_cluster(args, cfg)
        return
    if args.phase == "specialists":
        phase_specialists(args, phase_cluster(args, cfg))
        return
    if args.phase == "mix":
        manifest_path = out / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"no BTX manifest at {manifest_path}; run the specialists phase first")
        clusters = load_clusters(manifest_path)
        phase_mix(args, manifest_path, len(clusters))
        return

    clusters = phase_cluster(args, cfg)
    manifest_path = phase_specialists(args, clusters)
    phase_mix(args, manifest_path, len(clusters))


if __name__ == "__main__":
    main()
