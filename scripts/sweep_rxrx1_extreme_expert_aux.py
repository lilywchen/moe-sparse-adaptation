#!/usr/bin/env python
"""Extreme expert-count by router-auxiliary interaction screen.

The active low-temperature E32/E64 bracket tests whether a very large inactive expert bank changes
unseen-experiment transfer.  This bounded companion crosses E32 versus E64, canonical versus
within-experiment-balanced pressure, and tail-safe versus no auxiliary losses.  It tests whether
load balancing is specifically required to keep extreme expert banks usable, rather than treating
bank size alone as the mechanism.

Every arm fixes token top-1 routing, early placement, cosine geometry, temperature 0.03, seed 0,
data order, optimizer, and the 10/30/60 checkpoint schedule.  Active expert compute is matched;
total parameters intentionally differ between E32 and E64, so this family supports only an
active-compute-matched claim.  OOD test remains sealed, and every result is exploratory until a
locked fresh-seed comparison is run.
"""
import os
from pathlib import Path

from moe_shift.capacity.naming import run_id_from
from moe_shift.utils.config import apply_overrides, load_config
from scripts import sweep_rxrx1_expert_count as base
from scripts import sweep_rxrx1_router_aux as refill
from scripts.sweep_rxrx1_cell_dino import CONFIG


RESULT_ROOT = Path(os.environ.get(
    "MOE_RX_EXTREME_EXPERT_AUX_RESULTS",
    "/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/"
    "substrate_rxrx1/cell_dino_cp5/extreme_expert_aux60_20260802",
))
WANDB_GROUP = os.environ.get(
    "MOE_RX_WANDB_GROUP", "rxrx1-cell-dino-extreme-expert-aux60-20260802"
)
HF_PREFIX = os.environ.get(
    "MOE_RX_HF_PREFIX", "rxrx1/cell_dino_cp5/extreme_expert_aux60_20260802"
)
WANDB_JOB_TYPE = "rxrx1_extreme_expert_aux60"
WANDB_TAGS = (
    "rxrx1,cell-dino,extreme-expert-aux60,active-compute-matched,"
    "exploratory,ood-test-blind"
)

PRESSURES = ("canonical", "route")
EXPERT_COUNTS = (32, 64)
AUX_SETTINGS = (
    ("tail_safe", 1.0e-2, 1.0e-2),
    ("no_aux", 0.0, 0.0),
)


def cells(config=CONFIG):
    rows = []
    for pressure in PRESSURES:
        fixed = refill._base_overrides(config, pressure)
        for n_experts in EXPERT_COUNTS:
            for aux_label, balance_w, zloss_w in AUX_SETTINGS:
                label = f"{pressure}_E{n_experts}_{aux_label}"
                overrides = [
                    *fixed,
                    f"model.n_experts={n_experts}",
                    "model.temperature=0.03",
                    f"losses.balance_w={balance_w}",
                    f"losses.zloss_w={zloss_w}",
                    "analysis.run_mechanism=true",
                    "train.save_checkpoint_epochs=[10,30,60]",
                    f"run_tag=extreme_expert_aux60_{label}_20260802",
                ]
                cfg = apply_overrides(load_config(config), overrides)
                rows.append((label, overrides, run_id_from(cfg)))
    return rows


def main():
    base.RESULT_ROOT = RESULT_ROOT
    base.WANDB_GROUP = WANDB_GROUP
    base.HF_PREFIX = HF_PREFIX
    base.WANDB_JOB_TYPE = WANDB_JOB_TYPE
    base.WANDB_TAGS = WANDB_TAGS
    base.cells = cells
    base.main()


if __name__ == "__main__":
    main()
