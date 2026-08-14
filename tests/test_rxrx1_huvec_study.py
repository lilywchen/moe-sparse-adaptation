import pandas as pd
import pytest
import torch

from moe_shift.data.rxrx1_huvec import EXPECTED_TREATMENTS, deterministic_split
from moe_shift.models.huvec import MaskedAutoencoder, build_study_model
from scripts.run_rxrx1_huvec_study import _well_metrics
from scripts.sweep_rxrx1_huvec_study import planned_runs


def _site_frame():
    rows = []
    index = 0
    for experiment in range(24):
        for label in range(EXPECTED_TREATMENTS):
            well_id = f"{experiment}|0|{label}"
            for site in (1, 2):
                rows.append({
                    "global_index": index, "experiment": experiment, "label": label,
                    "well_id": well_id, "site": site, "relative_path": f"x/{well_id}_s{site}.png",
                })
                index += 1
    return pd.DataFrame(rows)


def test_custom_split_holds_out_experiments_and_never_splits_sites():
    frame = _site_frame()
    split = deterministic_split(frame, range(16), range(16, 24), "fold0")
    assert set(split[split.role == "target"].experiment) == set(range(16, 24))
    assert set(split[split.role != "target"].experiment) == set(range(16))
    assert split.groupby("well_id").role.nunique().max() == 1
    assert split.groupby("role").label.nunique().to_dict() == {
        "iid_validation": EXPECTED_TREATMENTS,
        "target": EXPECTED_TREATMENTS,
        "train": EXPECTED_TREATMENTS,
    }
    assert split[split.role == "iid_validation"].well_id.nunique() == 2 * EXPECTED_TREATMENTS


def test_custom_split_allows_small_audited_target_missingness_only():
    frame = _site_frame()
    incomplete_target = frame[~((frame.experiment == 16) & frame.label.isin(range(14)))].copy()
    split = deterministic_split(incomplete_target, range(16), [16], "missing_target")
    assert split[split.role == "target"].label.nunique() == EXPECTED_TREATMENTS - 14
    assert split[split.role == "train"].label.nunique() == EXPECTED_TREATMENTS
    assert split[split.role == "iid_validation"].label.nunique() == EXPECTED_TREATMENTS

    sparse_target = frame[~((frame.experiment == 16) & frame.label.isin(range(60)))].copy()
    with pytest.warns(RuntimeWarning, match="coverage recorded"):
        sparse_split = deterministic_split(
            sparse_target, range(16), [16], "sparse_but_nonempty_target")
    assert sparse_split[sparse_split.role == "target"].label.nunique() == (
        EXPECTED_TREATMENTS - 60)


def test_six_channel_models_and_parameter_match_are_shape_correct():
    images = torch.randn(2, 6, 32, 32)
    for kind in ("resnet18", "vit_tiny", "vit_tiny_moe", "vit_tiny_dense_matched"):
        model, audit = build_study_model(kind, num_classes=17, image_size=32)
        assert model(images).shape == (2, 17)
        assert audit["total_params"] > 0
    _, matched = build_study_model("vit_tiny_dense_matched", num_classes=17, image_size=32)
    assert matched["absolute_delta"] <= 2 * 192 + 2


def test_mae_uses_visible_tokens_and_reconstructs_six_channels():
    encoder, _ = build_study_model("vit_tiny_moe", num_classes=17, image_size=32)
    mae = MaskedAutoencoder(encoder, mask_ratio=0.75, decoder_dim=64,
                            decoder_depth=1, decoder_heads=4)
    reconstruction, auxiliary = mae(torch.randn(2, 6, 32, 32))
    assert reconstruction.ndim == 0 and torch.isfinite(reconstruction)
    assert auxiliary.ndim == 0 and torch.isfinite(auxiliary)


def test_top1_moe_router_receives_task_gradient():
    model, _ = build_study_model("vit_tiny_moe", num_classes=17, image_size=32)
    loss = torch.nn.functional.cross_entropy(model(torch.randn(2, 6, 32, 32)), torch.tensor([1, 2]))
    loss.backward()
    norms = []
    for block in model.moe_blocks:
        parameters = list(block.proj.parameters()) + [block.codebook, block.log_temp]
        norms.extend(float(parameter.grad.norm()) for parameter in parameters
                     if parameter.grad is not None)
    assert norms and all(torch.isfinite(torch.tensor(norms)))
    assert sum(norms) > 0


def test_well_metric_averages_two_site_logits_before_scoring():
    logits = torch.tensor([
        [4.0, 0.0], [0.0, 6.0],  # average predicts class 1 for well A
        [5.0, 0.0], [4.0, 0.0],  # predicts class 0 for well B
    ])
    labels = torch.tensor([1, 1, 0, 0])
    experiments = torch.tensor([9, 9, 10, 10])
    metrics, predictions = _well_metrics(logits, labels, experiments, ["A", "A", "B", "B"])
    assert metrics["n_wells"] == 2
    assert metrics["top1"] == 1.0
    assert set(predictions.n_sites) == {2}


def _registry():
    primary = [{
        "kind": "primary", "fold": fold, "split_id": f"primary_fold{fold}",
        "source_experiments": list(range(16)), "target_experiments": list(range(16, 24)),
        "difficulty_tier": "natural", "target_difficulty": {str(16 + fold): 0.1 + fold},
    } for fold in range(3)]
    controlled = []
    for target in (20, 21, 22):
        for index, tier in enumerate(("low", "medium", "high")):
            controlled.append({
                "kind": "controlled", "split_id": f"controlled_t{target}_{tier}_r0",
                "source_experiments": list(range(12)), "target_experiments": [target],
                "difficulty_tier": tier, "target_difficulty": {str(target): float(index)},
            })
    return {"main_training_splits": primary + controlled}


def test_one_seed_fast_registry_has_the_declared_43_runs():
    rows = planned_runs(_registry())
    assert len(rows) == 43
    assert len({row["run_id"] for row in rows}) == 43
    assert {stage: sum(row["stage"] == stage for row in rows)
            for stage in ("canary", "F_G", "H", "I", "J")} == {
                "canary": 2, "F_G": 13, "H": 23, "I": 3, "J": 2,
            }
    assert {row["seed"] for row in rows} == {0}
