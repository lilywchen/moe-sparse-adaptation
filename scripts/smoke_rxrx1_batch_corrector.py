#!/usr/bin/env python
"""One real native-RxRx1 GPU step for the batch-transport campaign."""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe_shift.capacity.model import build_ccas
from moe_shift.data import make_loaders, make_val_loader
from moe_shift.utils.config import apply_overrides, load_config
from scripts.run_ccas import cross_experiment_contrastive_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ccas_rxrx1_cell_dino_native.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), [
        "seed=9107", "model.variant=original", "model.batch_corrector.mode=moe_batch",
        "model.batch_corrector.n_experts=4", "model.batch_corrector.rank=16",
        "train.batch_size=16", "train.num_workers=2", "train.experiment_batching=true",
        "train.cross_experiment_pairs=true", "train.paired_experiment_batches=true",
    ])
    train, _within, _test, _audit = make_loaders(cfg)
    val = make_val_loader(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("the SciServer integration smoke test requires a GPU")
    model = build_ccas(cfg).to(device).train()
    batch = next(iter(train))
    environments = batch[3]
    unique_environments = torch.unique(environments)
    if len(unique_environments) != 2:
        raise RuntimeError(f"paired sampler emitted {len(unique_environments)} experiments")
    label_sets = [set(batch[1][environments == value].tolist()) for value in unique_environments]
    if label_sets[0] != label_sets[1]:
        raise RuntimeError("paired experiment halves do not have matched perturbations")
    x, y, site = batch[0].to(device), batch[1].to(device), batch[2].to(device)
    model.set_env(site); model.set_batch_environment(environments.to(device))
    features = model.forward_features(x)
    logits = model.fc(features)
    classification = F.cross_entropy(logits, y)
    alignment = cross_experiment_contrastive_loss(features, y, site, temperature=0.1)
    auxiliary = model.aux_loss(0.01, 0.001)
    loss = classification + 0.1 * alignment + auxiliary
    loss.backward()
    finite_gradients = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters())
    if not finite_gradients:
        raise RuntimeError("non-finite gradient in transport smoke test")
    model.eval()
    val_batch = next(iter(val))
    model.set_batch_environment(val_batch[3].to(device))
    with torch.no_grad():
        val_logits = model(val_batch[0].to(device))
    result = {
        "status": "passed", "gpu": torch.cuda.get_device_name(0),
        "paired_environments": unique_environments.tolist(),
        "matched_perturbations_per_environment": len(label_sets[0]),
        "feature_shape": list(features.shape), "logit_shape": list(logits.shape),
        "val_logit_shape": list(val_logits.shape),
        "classification_loss": float(classification), "alignment_loss": float(alignment),
        "auxiliary_loss": float(auxiliary), "total_loss": float(loss),
        "finite_gradients": finite_gradients,
        "corrector": dict(model.batch_corrector.last),
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
