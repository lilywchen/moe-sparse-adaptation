import hashlib
import json

import pandas as pd
import torch

from scripts.pretrain_rxrx1_huvec_mae import load_sealed_partition
from scripts.run_rxrx1_huvec_study import _make_loaders, _well_metrics


def _sha256(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _site(global_index, well_id, experiment, role, cell_type="HUVEC", label=0):
    return {
        "global_index": global_index, "well_id": well_id, "experiment": experiment,
        "experiment_name": f"{cell_type}-{experiment:02d}", "role": role,
        "cell_type": cell_type, "label": label, "site": 1,
        "relative_path": f"images/{cell_type}-{experiment:02d}/Plate1/{well_id}_s1.png",
    }


def test_full_loader_uses_checksum_frozen_assignment(tmp_path):
    rows = [
        _site(0, "A", 0, "train"), _site(1, "B", 0, "train"),
        _site(2, "C", 1, "iid_validation"),
        _site(3, "D", 2, "target", cell_type="RPE"),
    ]
    manifest = tmp_path / "treatment.parquet"
    assignment = tmp_path / "fold.parquet"
    pd.DataFrame(rows).drop(columns="role").to_parquet(manifest, index=False)
    pd.DataFrame(rows).to_parquet(assignment, index=False)
    registry = {"site_manifest": str(manifest), "raw_root": str(tmp_path)}
    split = {
        "split_id": "full_fold0", "source_experiments": [0, 1],
        "target_experiments": [2], "assignment": str(assignment),
        "assignment_sha256": _sha256(assignment),
        "normalization": {"mean": [0.0] * 6, "std": [1.0] * 6},
    }
    frozen, loaders = _make_loaders(
        tmp_path, registry, split, batch_size=2, workers=0, image_size=32,
        include_target=True, train_augmentation=False)
    assert frozen.set_index("well_id").role.to_dict() == {
        "A": "train", "B": "train", "C": "iid_validation", "D": "target"}
    assert len(loaders["train"].dataset) == 2
    assert len(loaders["iid_validation"].dataset) == 1
    assert len(loaders["target"].dataset) == 1


def test_full_mae_uses_source_controls_but_excludes_iid_and_target(tmp_path):
    assignment_rows = [
        _site(0, "A", 0, "train"), _site(1, "B", 0, "train"),
        _site(2, "C", 0, "iid_validation"),
        _site(4, "E", 1, "target", cell_type="RPE"),
    ]
    all_rows = [
        *[dict(row) for row in assignment_rows],
        _site(3, "D-control", 0, "train", label=1108),
        _site(5, "F-control", 1, "target", cell_type="RPE", label=1108),
    ]
    assignment = tmp_path / "fold.parquet"
    all_sites = tmp_path / "all.parquet"
    pd.DataFrame(assignment_rows).to_parquet(assignment, index=False)
    pd.DataFrame(all_rows).drop(columns="role").to_parquet(all_sites, index=False)
    registry = {
        "main_training_splits": [{
            "split_id": "full_fold0", "source_experiments": [0],
            "target_experiments": [1], "assignment": str(assignment),
            "assignment_sha256": _sha256(assignment),
            "physical_target_exclusion": False,
        }]
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry))
    _registry, _split, partition, audit = load_sealed_partition(
        registry_path, all_sites, "full_fold0", validation_fraction=0.34,
        source_experiment_count=1)
    assert set(partition.well_id) == {"A", "B", "D-control"}
    assert "C" not in set(partition.well_id)
    assert set(partition.experiment) == {0}
    assert audit["target_sites_excluded"] == 2
    assert audit["physical_target_exclusion_required"] is False


def test_well_metrics_report_cell_type_slices():
    logits = torch.tensor([[4.0, 0.0], [0.0, 4.0]])
    labels = torch.tensor([0, 1])
    experiments = torch.tensor([0, 1])
    metrics, predictions = _well_metrics(
        logits, labels, experiments, ["A", "B"], ["HUVEC", "RPE"])
    assert set(metrics["per_cell_type"]) == {"HUVEC", "RPE"}
    assert set(predictions.cell_type) == {"HUVEC", "RPE"}
    assert all(row["top1"] == 1.0 for row in metrics["per_cell_type"].values())
