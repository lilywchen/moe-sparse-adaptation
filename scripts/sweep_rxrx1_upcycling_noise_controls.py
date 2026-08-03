#!/usr/bin/env python
"""Learned-router, frozen-router, and dense controls for sparse upcycling noise.

The active ``upcycling_noise60`` family asks whether function-preserving expert noise creates
useful conditional specialization.  This bounded control bank separates that hypothesis from
ordinary parameter-noise regularization.  It reuses the already-declared learned-MoE anchors,
adds only the two missing learned dose anchors, and crosses frozen-router MoE and dense-wide
controls over E4/E8/E16 and noise 0.001/0.01.  OOD test remains sealed.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moe_shift.capacity.naming import run_id_from
from moe_shift.utils.config import apply_overrides, load_config
from scripts import sweep_rxrx1_expert_count as base
from scripts import sweep_rxrx1_router_aux as refill
from scripts.sweep_rxrx1_cell_dino import CONFIG


RESULT_ROOT = Path(os.environ.get(
    "MOE_RX_UPCYCLING_NOISE_CONTROL_RESULTS",
    "/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/"
    "substrate_rxrx1/cell_dino_cp5/upcycling_noise_controls60_20260803",
))
WANDB_GROUP = os.environ.get(
    "MOE_RX_WANDB_GROUP", "rxrx1-cell-dino-upcycling-noise-controls60-20260803"
)
HF_PREFIX = os.environ.get(
    "MOE_RX_HF_PREFIX", "rxrx1/cell_dino_cp5/upcycling_noise_controls60_20260803"
)
WANDB_JOB_TYPE = "rxrx1_upcycling_noise_controls60"
WANDB_TAGS = (
    "rxrx1,cell-dino,upcycling-noise-controls60,mechanism-control,exploratory,"
    "ood-test-blind"
)

TEMPERATURE = 0.3
BALANCE_W = 1.0e-2
ZLOSS_W = 1.0e-2
EXPERT_COUNTS = (4, 8, 16)
NOISE_LEVELS = (1.0e-3, 1.0e-2)

# Learned anchors already present in upcycling_noise60 and therefore never duplicated here.
SHARED_LEARNED = {
    (4, 1.0e-2): "canonical_E4_temp03_noise0p01_tail_safe",
    (8, 1.0e-3): "canonical_E8_temp03_noise0p001_tail_safe",
    (8, 1.0e-2): "canonical_E8_temp03_noise0p01_tail_safe",
    (16, 1.0e-2): "canonical_E16_temp03_noise0p01_tail_safe",
}


def _noise_label(noise):
    return str(noise).replace(".", "p").replace("-", "m")


def cells(config=CONFIG):
    rows = []
    fixed = refill._base_overrides(config, "canonical")
    for n_experts in EXPERT_COUNTS:
        for noise in NOISE_LEVELS:
            noise_label = _noise_label(noise)
            learned_label = f"canonical_E{n_experts}_temp03_noise{noise_label}_tail_safe"

            # Add only learned dose anchors that are not already active/queued elsewhere.
            if (n_experts, noise) not in SHARED_LEARNED:
                label = f"learned_E{n_experts}_temp03_noise{noise_label}_tail_safe"
                overrides = [
                    *fixed,
                    "model.variant=moe",
                    f"model.n_experts={n_experts}",
                    f"model.temperature={TEMPERATURE}",
                    f"model.sym_break_moe={noise}",
                    f"losses.balance_w={BALANCE_W}",
                    f"losses.zloss_w={ZLOSS_W}",
                    "analysis.run_mechanism=true",
                    "train.save_checkpoint_epochs=[10,30,60]",
                    f"run_tag=upcycling_noise_controls60_{label}_20260803",
                ]
                cfg = apply_overrides(load_config(config), overrides)
                rows.append((label, overrides, run_id_from(cfg)))

            frozen_label = f"frozen_E{n_experts}_temp03_noise{noise_label}_tail_safe"
            frozen = [
                *fixed,
                "model.variant=moe_frozen",
                f"model.n_experts={n_experts}",
                f"model.temperature={TEMPERATURE}",
                f"model.sym_break_moe={noise}",
                f"losses.balance_w={BALANCE_W}",
                f"losses.zloss_w={ZLOSS_W}",
                "analysis.run_mechanism=true",
                "train.save_checkpoint_epochs=[10,30,60]",
                f"run_tag=upcycling_noise_controls60_{frozen_label}_20260803",
            ]
            cfg = apply_overrides(load_config(config), frozen)
            rows.append((frozen_label, frozen, run_id_from(cfg)))

            dense_label = f"dense_E{n_experts}_temp03_noise{noise_label}"
            dense = [
                *fixed,
                "model.variant=dense_wide",
                f"model.n_experts={n_experts}",
                f"model.temperature={TEMPERATURE}",
                "model.sym_break_moe=0.0",
                f"model.sym_break_wide={noise}",
                f"losses.balance_w={BALANCE_W}",
                f"losses.zloss_w={ZLOSS_W}",
                "analysis.run_mechanism=true",
                "train.save_checkpoint_epochs=[10,30,60]",
                f"run_tag=upcycling_noise_controls60_{dense_label}_20260803",
            ]
            cfg = apply_overrides(load_config(config), dense)
            rows.append((dense_label, dense, run_id_from(cfg)))

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
