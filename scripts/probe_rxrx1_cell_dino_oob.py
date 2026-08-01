#!/usr/bin/env python
"""Frozen Cell-DINO representation probe on RxRx1.

Cell-DINO is self-supervised and has no RxRx1 classifier head.  The closest genuinely
out-of-the-box test is therefore a non-parametric readout of the frozen embeddings.  This script
extracts one train-split embedding per image and evaluates two deterministic classifiers:

* exact cosine 1-nearest-neighbour (the standard DINO representation probe);
* cosine nearest class centroid, which averages each perturbation across training batches.

Only ID-test and OOD-validation are evaluated.  The WILDS OOD-test loader is never iterated.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe_shift.capacity.model import build_ccas
from moe_shift.data import make_loaders, make_val_loader
from moe_shift.utils.config import apply_overrides, load_config


def _git_info():
    sha = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip())
    return sha, dirty


@torch.inference_mode()
def extract(model, loader, device):
    features, labels, environments = [], [], []
    model.eval()
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        z = F.normalize(model.forward_features(x).float(), dim=1)
        features.append(z.cpu())
        labels.append(batch[1].long().cpu())
        environments.append((batch[3] if len(batch) > 3 else batch[2]).long().cpu())
    return torch.cat(features), torch.cat(labels), torch.cat(environments)


def class_centroids(features, labels, n_classes):
    sums = features.new_zeros((n_classes, features.shape[1]))
    counts = features.new_zeros(n_classes)
    sums.index_add_(0, labels, features)
    counts.index_add_(0, labels, torch.ones_like(labels, dtype=features.dtype))
    if torch.any(counts == 0):
        missing = torch.nonzero(counts == 0).flatten().tolist()
        raise RuntimeError(f"training split is missing {len(missing)} classes: {missing[:10]}")
    return F.normalize(sums / counts[:, None], dim=1)


def summarize(pred, labels, environments):
    correct = pred.eq(labels)
    per_env = {}
    per_env_n = {}
    for env in torch.unique(environments).tolist():
        mask = environments.eq(env)
        per_env[str(int(env))] = float(correct[mask].float().mean())
        per_env_n[str(int(env))] = int(mask.sum())
    return {
        "accuracy": float(correct.float().mean()),
        "worst_env_accuracy": min(per_env.values()),
        "per_env_accuracy": per_env,
        "per_env_n": per_env_n,
    }


@torch.inference_mode()
def evaluate_readouts(train_features, train_labels, query_features, query_labels,
                      query_environments, n_classes, device, query_chunk=256):
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    centroids = class_centroids(train_features, train_labels, n_classes)
    knn_pred, centroid_pred = [], []
    for start in range(0, len(query_features), query_chunk):
        q = query_features[start:start + query_chunk].to(device)
        knn_idx = (q @ train_features.T).argmax(dim=1)
        knn_pred.append(train_labels[knn_idx].cpu())
        centroid_pred.append((q @ centroids.T).argmax(dim=1).cpu())
    return {
        "cosine_1nn": summarize(torch.cat(knn_pred), query_labels, query_environments),
        "nearest_centroid": summarize(
            torch.cat(centroid_pred), query_labels, query_environments
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ccas_rxrx1_cell_dino.yaml")
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rxrx1_cell_dino_frozen_oob_readouts_s0.json"
    if out_path.exists():
        print(f"[skip] {out_path} already exists")
        return

    cfg = apply_overrides(load_config(args.config), args.override)
    cfg["model"]["variant"] = "original"
    cfg["model"]["freeze_backbone"] = True
    cfg["seed"] = 0
    if int(cfg.get("stage", 1)) >= 3:
        raise ValueError("out-of-box diagnosis must remain pre-confirmatory")

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()

    train_loader, id_loader, _unused_ood_test_loader, _audit_loader = make_loaders(cfg)
    val_loader = make_val_loader(cfg)
    if val_loader is None:
        raise RuntimeError("RxRx1 OOD-validation split is required")
    model = build_ccas(cfg).to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    train_f, train_y, _ = extract(model, train_loader, device)
    id_f, id_y, id_env = extract(model, id_loader, device)
    val_f, val_y, val_env = extract(model, val_loader, device)
    n_classes = int(cfg["model"]["num_classes"])
    metrics = {
        "id_test": evaluate_readouts(
            train_f, train_y, id_f, id_y, id_env, n_classes, device
        ),
        "ood_val": evaluate_readouts(
            train_f, train_y, val_f, val_y, val_env, n_classes, device
        ),
    }
    sha, dirty = _git_info()
    result = {
        "run_id": out_path.stem,
        "dataset": "rxrx1",
        "seed": 0,
        "probe": "frozen_nonparametric",
        "readouts": ["cosine_1nn", "nearest_centroid"],
        "selection_split": "ood_val",
        "test_evaluated": False,
        "git_sha": sha,
        "git_dirty": dirty,
        "checkpoint_provenance": model.backbone_provenance,
        "n_train_embeddings": int(len(train_f)),
        "embedding_dim": int(train_f.shape[1]),
        "metrics": metrics,
        "elapsed_seconds": round(time.time() - started, 1),
        "config": cfg,
        "environment": {"hostname": os.uname().nodename, "torch": torch.__version__},
    }
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    tmp.replace(out_path)
    print(json.dumps({
        "result": str(out_path),
        "id_1nn": metrics["id_test"]["cosine_1nn"]["accuracy"],
        "ood_val_1nn": metrics["ood_val"]["cosine_1nn"]["accuracy"],
        "id_centroid": metrics["id_test"]["nearest_centroid"]["accuracy"],
        "ood_val_centroid": metrics["ood_val"]["nearest_centroid"]["accuracy"],
        "test_evaluated": False,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
