#!/usr/bin/env python3
"""One-command real-data smoke and throughput gate for the RxRx3-core loader."""

import argparse
import json
import time

import torch

from moe_shift.data import make_loaders, make_val_loader
from moe_shift.utils.config import apply_overrides, load_config


def inspect_loader(name, loader, batches):
    started = time.time()
    rows = 0
    batches_seen = 0
    shapes, labels, sites, environments = set(), set(), set(), set()
    for index, batch in enumerate(loader):
        x, y, site, environment = batch[:4]
        if not torch.isfinite(x).all():
            raise ValueError(f"{name} contains non-finite pixels")
        if x.ndim != 4 or x.shape[1] != 5:
            raise ValueError(f"{name} expected Bx5xHxW, got {tuple(x.shape)}")
        rows += int(y.numel())
        batches_seen += 1
        shapes.add(tuple(x.shape[1:]))
        labels.update(map(int, y.tolist()))
        sites.update(map(int, site.tolist()))
        environments.update(map(int, environment.tolist()))
        if index + 1 >= batches:
            break
    elapsed = time.time() - started
    return {
        "batches": batches_seen,
        "rows": rows,
        "shapes": [list(value) for value in sorted(shapes)],
        "labels_observed": len(labels),
        "sites_observed": sorted(sites),
        "environments_observed": sorted(environments),
        "seconds": round(elapsed, 3),
        "rows_per_second": round(rows / max(elapsed, 1e-9), 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ccas_rxrx3_core_cell_dino.yaml")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument("--batches", type=int, default=4)
    args = parser.parse_args()
    if args.batches < 1:
        raise ValueError("--batches must be positive")
    cfg = apply_overrides(load_config(args.config), args.override)
    train, id_val, ood_test, audit = make_loaders(cfg)
    validation = make_val_loader(cfg)
    if audit is not id_val or validation is not id_val:
        raise ValueError("RxRx3 audit/selection loaders must reuse the frozen ID validation split")
    report = {
        "dataset": cfg["dataset"],
        "manifest": cfg["rxrx3_manifest"],
        "sites": cfg["sites"],
        "selection_split": validation.selection_split_name,
        "dataset_rows": {
            "train": len(train.dataset), "id_val": len(id_val.dataset),
            "ood_test": len(ood_test.dataset),
        },
        "observed": {
            "train": inspect_loader("train", train, args.batches),
            "id_val": inspect_loader("id_val", id_val, args.batches),
            "ood_test": inspect_loader("ood_test", ood_test, args.batches),
        },
    }
    if any(site < 0 for site in report["observed"]["train"]["sites_observed"]):
        raise ValueError("training rows contain an unseen-site sentinel")
    if report["observed"]["ood_test"]["sites_observed"] != [-1]:
        raise ValueError("OOD-test rows must all use site=-1")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
