#!/usr/bin/env python
"""Locked fresh-seed confirmation of the tail-safe router-auxiliary survivor.

The complete seed-0 router-auxiliary screen selected the canonical early token-cosine E8 top-1
configuration with balance weight 0.01 and router-z-loss weight 0.01.  At epoch 60 it improves
mean OOD validation and worst-experiment accuracy over the exact same-placement dense-wide control,
which licenses the predeclared smaller-mean/consistent-tail confirmation path.  This registry locks
that sparse configuration and its exact dense comparator at fresh seeds 1 and 2.  No additional
tuning is permitted from these runs.

All four arms use the same data order per seed, optimizer, 60-epoch horizon, and 10/30/60
checkpoints.  The sparse-versus-dense claim is exact-total-parameter matched up to the already
validated 378-parameter (0.001232%) implementation difference.  OOD validation selects only the
predeclared milestone interpretation; OOD test remains sealed.  A replicated material effect may
license a separately frozen 90-epoch adjudication, but these jobs do not evaluate test.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe_shift.capacity.naming import run_id_from
from moe_shift.utils.config import apply_overrides, load_config
from scripts import sweep_rxrx1_router_aux as refill
from scripts.sweep_rxrx1_cell_dino import CONFIG, cells as factorial_cells


RESULT_ROOT = Path(os.environ.get(
    "MOE_RX_TAIL_SAFE_CONFIRM_RESULTS",
    "/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/"
    "substrate_rxrx1/cell_dino_cp5/tail_safe_confirm60_20260802",
))
WANDB_GROUP = os.environ.get(
    "MOE_RX_WANDB_GROUP", "rxrx1-cell-dino-tail-safe-confirm60-20260802"
)
HF_PREFIX = os.environ.get(
    "MOE_RX_HF_PREFIX", "rxrx1/cell_dino_cp5/tail_safe_confirm60_20260802"
)

SEEDS = (1, 2)
BALANCE_W = 1.0e-2
ZLOSS_W = 1.0e-2


def _dense_base(config):
    for tag, overrides, _ in factorial_cells(config):
        if tag == "dense_early_canonical":
            return list(overrides)
    raise RuntimeError("missing exact early-canonical dense-wide control")


def cells(config=CONFIG):
    rows = []
    sparse_base = refill._base_overrides(config, "canonical")
    dense_base = _dense_base(config)
    for seed in SEEDS:
        sparse_overrides = [
            *sparse_base,
            f"seed={seed}",
            f"losses.balance_w={BALANCE_W}",
            f"losses.zloss_w={ZLOSS_W}",
            "analysis.run_mechanism=true",
            "train.save_checkpoint_epochs=[10,30,60]",
            f"run_tag=tail_safe_confirm60_moe_s{seed}_20260802",
        ]
        sparse_cfg = apply_overrides(load_config(config), sparse_overrides)
        rows.append((f"moe_s{seed}", sparse_overrides, run_id_from(sparse_cfg)))

        dense_overrides = [
            *dense_base,
            f"seed={seed}",
            "analysis.run_mechanism=true",
            "train.save_checkpoint_epochs=[10,30,60]",
            f"run_tag=tail_safe_confirm60_dense_s{seed}_20260802",
        ]
        dense_cfg = apply_overrides(load_config(config), dense_overrides)
        rows.append((f"dense_s{seed}", dense_overrides, run_id_from(dense_cfg)))
    return rows


def main():
    refill.RESULT_ROOT = RESULT_ROOT
    refill.WANDB_GROUP = WANDB_GROUP
    refill.HF_PREFIX = HF_PREFIX
    refill.cells = cells
    refill.main()


if __name__ == "__main__":
    main()
