import json

import pandas as pd
import pytest
import torch

from moe_shift.data.rxrx1_huvec import EXPECTED_TREATMENTS, deterministic_split
from moe_shift.models.huvec import MaskedAutoencoder, build_study_model
from scripts.aggregate_rxrx1_huvec_study import _target_rows
from scripts.certify_rxrx1_huvec_recipe import (
    _curve_history,
    _plateau_history,
    choose_source_iid_checkpoint,
    default_recipes,
    format_status,
)
from scripts.evaluate_rxrx1_huvec_sites import _agreement_summary, _site_metrics
from scripts.launch_rxrx1_huvec_recipe_pair import _status_signature, pair_plan
from scripts.prepare_rxrx1_huvec_study import _split_arrays
from scripts.run_rxrx1_huvec_study import _well_metrics
from scripts.pretrain_rxrx1_huvec_mae import load_sealed_partition, partition_mae_wells
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


def test_probe_split_accepts_the_cached_well_level_manifest_without_site_column():
    well_frame = (_site_frame().drop_duplicates("well_id")
                  [["well_id", "experiment", "label"]].copy())
    well_frame["n_sites"] = 2
    features = torch.zeros((len(well_frame), 4)).numpy()
    metadata, indices = _split_arrays(well_frame, features, {
        "source_experiments": list(range(16)),
        "target_experiments": list(range(16, 24)),
        "split_id": "well_level_probe",
    })
    assert "site" not in metadata.columns
    assert len(metadata) == len(well_frame)
    assert {role: len(selected) for role, selected in indices.items()} == {
        "train": 14 * EXPECTED_TREATMENTS,
        "iid_validation": 2 * EXPECTED_TREATMENTS,
        "target": 8 * EXPECTED_TREATMENTS,
    }


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
    for kind in (
        "resnet18", "vit_micro", "vit_tiny", "vit_tiny_moe",
        "vit_tiny_dense_matched",
    ):
        model, audit = build_study_model(kind, num_classes=17, image_size=32)
        assert model(images).shape == (2, 17)
        assert audit["total_params"] > 0
    _, matched = build_study_model("vit_tiny_dense_matched", num_classes=17, image_size=32)
    assert matched["absolute_delta"] <= 2 * 192 + 2
    _, micro = build_study_model("vit_micro", num_classes=1108, image_size=224)
    _, tiny = build_study_model("vit_tiny", num_classes=1108, image_size=224)
    assert micro["total_params"] == 1_554_900
    assert micro["total_params"] < tiny["total_params"]


def test_mae_uses_visible_tokens_and_reconstructs_six_channels():
    encoder, _ = build_study_model("vit_tiny_moe", num_classes=17, image_size=32)
    mae = MaskedAutoencoder(encoder, mask_ratio=0.75, decoder_dim=64,
                            decoder_depth=1, decoder_heads=4)
    reconstruction, auxiliary = mae(torch.randn(2, 6, 32, 32))
    assert reconstruction.ndim == 0 and torch.isfinite(reconstruction)
    assert auxiliary.ndim == 0 and torch.isfinite(auxiliary)


def test_mae_partition_is_deterministic_and_keeps_sites_with_their_well():
    rows = []
    for experiment in (11, 12, 13):
        for well in range(20):
            for site in (1, 2):
                rows.append({
                    "global_index": len(rows), "experiment": experiment,
                    "well_id": f"{experiment}|P1|W{well:02d}", "site": site,
                })
    frame = pd.DataFrame(rows)
    first = partition_mae_wells(frame, validation_fraction=0.10, seed=7)
    second = partition_mae_wells(frame, validation_fraction=0.10, seed=7)
    assert first[["global_index", "mae_role"]].equals(
        second[["global_index", "mae_role"]])
    assert first.groupby("well_id").mae_role.nunique().max() == 1
    validation = first[first.mae_role == "mae_validation"]
    assert validation.groupby("experiment").well_id.nunique().to_dict() == {
        11: 2, 12: 2, 13: 2,
    }


def test_mae_source_scaling_uses_a_nested_centrality_order(tmp_path):
    frame = _site_frame()
    manifest = tmp_path / "sites.parquet"
    registry = tmp_path / "registry.json"
    frame.to_parquet(manifest, index=False)
    registry.write_text(json.dumps({
        "centrality": {str(value): float(value) for value in range(24)},
        "main_training_splits": [{
            "split_id": "fold0", "source_experiments": list(range(16)),
            "target_experiments": list(range(16, 24)),
            "normalization": {"mean": [0.0] * 6, "std": [1.0] * 6},
        }],
    }))
    _, _, four, audit4 = load_sealed_partition(
        registry, manifest, "fold0", source_experiment_count=4)
    _, _, eight, audit8 = load_sealed_partition(
        registry, manifest, "fold0", source_experiment_count=8)
    assert set(four.experiment.unique()) == {0, 1, 2, 3}
    assert set(four.experiment.unique()) < set(eight.experiment.unique())
    assert audit4["nested_source_experiment_order"] == audit8[
        "nested_source_experiment_order"]


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
    assert metrics["site_top1"] == 0.75
    assert set(predictions.n_sites) == {2}


def test_recipe_checkpoint_audit_tracks_best_source_iid_without_a_train_gate():
    earlier = {"epoch": 40, "selection_train_top1": 0.81, "selection_iid_top1": 0.07}
    later_better = {"epoch": 50, "selection_train_top1": 0.20, "selection_iid_top1": 0.08}
    later_tie = {"epoch": 60, "selection_train_top1": 0.95, "selection_iid_top1": 0.08}
    assert choose_source_iid_checkpoint(earlier, later_better) is later_better
    assert choose_source_iid_checkpoint(later_better, later_tie) is later_better


def test_recipe_curve_recovery_keeps_low_train_high_iid_checkpoint(tmp_path):
    curve = tmp_path / "curves.jsonl"
    rows = [
        {
            "phase": "supervised", "epoch": 5, "evaluated": True,
            "train_augmented_loss": 5.0, "train_augmented_site_top1": 0.1,
            "learning_rate": 1e-3, "selection_train_top1": 0.2,
            "selection_iid_top1": 0.3, "train_site_top1": 0.2,
            "train_well_top1": 0.3, "train_site_loss": 4.0,
            "train_well_loss": 3.0, "iid_site_top1": 0.3,
            "iid_well_top1": 0.4, "iid_site_loss": 3.5, "iid_well_loss": 3.0,
            "eligible": False,
        },
        {
            "phase": "supervised", "epoch": 10, "evaluated": True,
            "train_augmented_loss": 1.0, "train_augmented_site_top1": 0.9,
            "learning_rate": 5e-4, "selection_train_top1": 0.9,
            "selection_iid_top1": 0.25, "train_site_top1": 0.9,
            "train_well_top1": 0.95, "train_site_loss": 0.5,
            "train_well_loss": 0.4, "iid_site_top1": 0.25,
            "iid_well_top1": 0.35, "iid_site_loss": 3.7, "iid_well_loss": 3.2,
            "threshold_reached": True,
        },
    ]
    curve.write_text("".join(f"{json.dumps(row)}\n" for row in rows))
    augmented, latest, best, threshold = _curve_history(curve)
    assert augmented["site_top1"] == 0.9
    assert latest["epoch"] == 10
    assert best["epoch"] == 5
    assert threshold is True
    plateau_score, plateau_epoch, stale = _plateau_history(curve, min_delta=0.001)
    assert plateau_score == 0.3
    assert plateau_epoch == 5
    assert stale == 1


def test_plateau_history_resets_only_for_meaningful_source_iid_improvement(tmp_path):
    curve = tmp_path / "curves.jsonl"
    scores = [0.2000, 0.2005, 0.2011, 0.2009, 0.2008]
    rows = [
        {
            "phase": "supervised", "epoch": 5 * (index + 1),
            "evaluated": True, "selection_iid_top1": score,
        }
        for index, score in enumerate(scores)
    ]
    curve.write_text("".join(f"{json.dumps(row)}\n" for row in rows))
    score, epoch, stale = _plateau_history(curve, min_delta=0.001)
    assert score == 0.2011
    assert epoch == 15
    assert stale == 2


def test_recipe_ladder_is_bounded_regularized_and_live_status_is_interpretable():
    recipes = default_recipes("resnet18")
    assert recipes and all(recipe["augmentation"] for recipe in recipes)
    assert all(recipe["max_epochs"] > 30 for recipe in recipes)
    rendered = format_status({
        "state": "training", "model": "resnet18", "split_id": "primary_fold0",
        "attempt_index": 1, "n_attempts": len(recipes), "attempt_name": recipes[0]["name"],
        "epoch": 5, "max_epochs": recipes[0]["max_epochs"], "train_unit": "site",
        "train_threshold": 0.8,
        "latest_augmented": {"site_top1": 0.1, "loss": 5.0, "learning_rate": 1e-3},
        "latest_evaluation": {
            "train_site_top1": 0.2, "train_well_top1": 0.3,
            "iid_site_top1": 0.04, "iid_well_top1": 0.05, "eligible": False,
        },
    })
    assert "full unaugmented train: site=0.2000 well=0.3000" in rendered
    assert "target batches: excluded" in rendered
    complete = format_status({
        "state": "complete", "model": "resnet18", "split_id": "primary_fold0",
        "attempt_index": 1, "n_attempts": 1, "attempt_name": "standard",
        "epoch": 120, "recipe": {"max_epochs": 120}, "elapsed_seconds": 600,
    })
    assert "epoch=120/120" in complete
    assert "wall clock: 10.0 minutes" in complete


def test_two_gpu_recipe_pair_launcher_covers_all_six_candidates_without_collisions():
    plans = [pair_plan(index) for index in range(3)]
    assert all([item["gpu"] for item in plan] == ["0", "1"] for plan in plans)
    assert all([item["model"] for item in plan] == ["vit_micro", "vit_tiny"]
               for plan in plans)
    assert len({(item["run_name"], item["model"])
                for plan in plans for item in plan}) == 6
    assert [plan[0]["recipe"]["name"] for plan in plans] == [
        "adamw_standard_extended", "adamw_low_regularization",
        "adamw_weak_regularization",
    ]


def test_recipe_monitor_signature_changes_only_when_persisted_status_changes():
    payload = {
        "state": "training", "attempt_name": "standard", "epoch": 5,
        "updated_at": 100.0,
    }
    assert _status_signature(payload) == _status_signature(dict(payload))
    assert _status_signature(payload) != _status_signature({**payload, "epoch": 6})
    assert _status_signature(None) is None


def test_site_metrics_keep_both_fields_and_audit_mean_logit_pooling():
    logits = torch.zeros(4, EXPECTED_TREATMENTS)
    logits[0, 1] = 5.0  # well A site 1 correct
    logits[1, 0] = 5.0  # well A site 2 incorrect
    logits[2, 0] = 5.0  # well B site 1 correct
    logits[3, 0] = 4.0  # well B site 2 correct
    metrics, sites = _site_metrics(
        logits, torch.tensor([1, 1, 0, 0]), torch.tensor([9, 9, 10, 10]),
        ["A", "A", "B", "B"], torch.tensor([1, 2, 1, 2]), torch.arange(4))
    wells = pd.DataFrame({
        "well_id": ["A", "B"], "correct_top1": [True, True],
    })
    agreement = _agreement_summary(sites, wells)
    assert metrics["n_sites"] == 4 and metrics["n_wells"] == 2
    assert metrics["top1"] == 0.75
    assert metrics["per_experiment"]["9"]["top1"] == 0.5
    assert agreement["two_site_prediction_agreement"] == 0.5
    assert agreement["both_sites_correct"] == 0.5
    assert agreement["exactly_one_site_correct"] == 0.5
    assert agreement["well_top1_on_two_site_wells"] == 1.0


def test_aggregator_joins_site_and_well_target_metrics_by_experiment():
    result = {
        "run_id": "example", "model": "vit_tiny", "split_id": "primary_fold0",
        "split_kind": "primary", "difficulty_tier": "natural",
        "target": {"per_experiment": {
            "5": {"top1": 0.3, "top5": 0.5, "mean_rank": 12.0},
        }},
        "train": {"top1": 0.8}, "iid_validation": {"top1": 0.4},
        "target_difficulty": {"5": 0.1}, "raw_qc_target_difficulty": {"5": 0.2},
        "target_label_coverage": {"5": {
            "observed_labels": EXPECTED_TREATMENTS,
            "source_matched_labels": EXPECTED_TREATMENTS, "fraction": 1.0,
        }},
        "role_label_coverage": {
            "train": {"fraction": 1.0}, "iid_validation": {"fraction": 1.0},
        },
        "training_certified": True, "model_audit": {"total_params": 10}, "best_epoch": 5,
    }
    site_result = {"roles": {
        "train": {"top1": 0.75}, "iid_validation": {"top1": 0.35},
        "target": {
            "per_experiment": {"5": {"top1": 0.25, "top5": 0.45, "mean_rank": 15.0}},
            "agreement": {"per_experiment": {"5": {
                "two_site_prediction_agreement": 0.6,
                "both_sites_correct": 0.2, "exactly_one_site_correct": 0.1,
                "neither_site_correct": 0.7,
                "well_correct_when_neither_site_correct": 0.02,
                "well_incorrect_when_at_least_one_site_correct": 0.01,
            }}},
        },
    }}
    target = _target_rows([{
        "spec": {"stage": "H"}, "result": result, "site_result": site_result,
    }]).iloc[0]
    assert target.site_target_top1 == 0.25
    assert target.well_minus_site_target_top1 == pytest.approx(0.05)
    assert target.site_iid_to_target_gap == pytest.approx(0.10)
    assert target.two_site_prediction_agreement == 0.6


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
