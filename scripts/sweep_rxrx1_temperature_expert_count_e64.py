#!/usr/bin/env python
"""Extreme overfragmentation bracket for the low-temperature expert-count screen.

This bounded extension adds canonical- and route-pressure E64 cells without changing the
E2/E4/E16 or E32 registries and their shard identities. The completed E8 pair remains the shared
anchor. Both seed-0 jobs retain top-1 early token-cosine routing, temperature 0.03, zero load-
balance loss, router z-loss 0.001, and 10/30/60 checkpoints. Active expert compute is approximately
matched while total parameters intentionally increase, so this is an exploratory active-compute-
matched overfragmentation falsifier only; OOD test remains sealed.
"""
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import sweep_rxrx1_temperature_expert_count as base


WANDB_JOB_TYPE = "rxrx1_temperature_expert_count60_e64"
WANDB_TAGS = (
    "rxrx1,cell-dino,temperature-expert-count60,e64-overfragmentation,exploratory,"
    "active-compute-matched,ood-test-blind"
)


def cells(config=base.CONFIG):
    previous = base.EXPERT_COUNTS
    base.EXPERT_COUNTS = (64,)
    try:
        return base.cells(config)
    finally:
        base.EXPERT_COUNTS = previous


def main():
    previous_counts = base.EXPERT_COUNTS
    previous_job_type = base.WANDB_JOB_TYPE
    previous_tags = base.WANDB_TAGS
    base.EXPERT_COUNTS = (64,)
    base.WANDB_JOB_TYPE = WANDB_JOB_TYPE
    base.WANDB_TAGS = WANDB_TAGS
    try:
        base.main()
    finally:
        base.EXPERT_COUNTS = previous_counts
        base.WANDB_JOB_TYPE = previous_job_type
        base.WANDB_TAGS = previous_tags


if __name__ == "__main__":
    main()
