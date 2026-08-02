#!/usr/bin/env python
"""Active-compute-matched expert-count screen for the early token-cosine MoE.

The completed router-auxiliary screen identifies a tail-safe canonical configuration with high
load-balance and router-z losses, while the fixed-temperature epoch-10 screen is nearly flat over
the tested initial temperatures.  This bounded follow-up asks whether the number of available
experts, rather than initial router sharpness, controls specialization and unseen-experiment
robustness.

The screen crosses 4 versus 16 experts, canonical versus within-experiment route pressure, and the
tail-safe auxiliary setting versus a no-auxiliary control.  Expert width, top-1 activation, data
order, optimizer, and 10/30/60 checkpoints remain fixed.  Active expert compute is therefore
matched, but total parameters intentionally vary with expert count; this family cannot support an
exact-total-parameter or original-budget claim.  The shared early dense-wide result is only the
active-compute reference.  Every result is exploratory until a locked configuration is confirmed
with fresh seeds, and OOD test remains sealed.
"""
import os
from pathlib import Path

from moe_shift.capacity.naming import run_id_from
from moe_shift.utils.config import apply_overrides, load_config
from scripts import sweep_rxrx1_router_aux as refill
from scripts.sweep_rxrx1_cell_dino import CONFIG


RESULT_ROOT = Path(os.environ.get(
    "MOE_RX_EXPERT_COUNT_RESULTS",
    "/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/"
    "substrate_rxrx1/cell_dino_cp5/expert_count60_20260802",
))
WANDB_GROUP = os.environ.get(
    "MOE_RX_WANDB_GROUP", "rxrx1-cell-dino-expert-count60-20260802"
)
HF_PREFIX = os.environ.get(
    "MOE_RX_HF_PREFIX", "rxrx1/cell_dino_cp5/expert_count60_20260802"
)

PRESSURES = ("route", "canonical")
EXPERT_COUNTS = (4, 16)
AUX_SETTINGS = (
    ("tail_safe", 1.0e-2, 1.0e-2),
    ("no_aux", 0.0, 0.0),
)


def cells(config=CONFIG):
    rows = []
    for pressure in PRESSURES:
        base = refill._base_overrides(config, pressure)
        for n_experts in EXPERT_COUNTS:
            for aux_label, balance_w, zloss_w in AUX_SETTINGS:
                label = f"{pressure}_E{n_experts}_{aux_label}"
                overrides = [
                    *base,
                    f"model.n_experts={n_experts}",
                    f"losses.balance_w={balance_w}",
                    f"losses.zloss_w={zloss_w}",
                    "analysis.run_mechanism=true",
                    "train.save_checkpoint_epochs=[10,30,60]",
                    f"run_tag=expert_count60_{label}_20260802",
                ]
                cfg = apply_overrides(load_config(config), overrides)
                rows.append((label, overrides, run_id_from(cfg)))
    return rows


def main():
    refill.RESULT_ROOT = RESULT_ROOT
    refill.WANDB_GROUP = WANDB_GROUP
    refill.HF_PREFIX = HF_PREFIX
    refill.cells = cells
    refill.main()


if __name__ == "__main__":
    main()
