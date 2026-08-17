from pathlib import Path

import numpy as np
import pandas as pd
import torch

from moe_shift.models.rxrx1_calibration import (
    ProfilingHead,
    build_rxrx1_calibration_model,
)
from scripts.freeze_rxrx1_huvec_crossfits import freeze
from scripts.train_rxrx1_calibration import EvenTrainSampler, cutmix


def test_profiling_head_contract():
    head = ProfilingHead(32, num_classes=11, embedding_dim=16)
    head.eval()
    x = torch.randn(3, 32)
    assert head.features(x).shape == (3, 16)
    assert head(x).shape == (3, 11)


def test_resnet50_six_channel_contract():
    model = build_rxrx1_calibration_model(
        "resnet50", num_classes=11, image_size=64, pretrained=False)
    model.eval()
    with torch.inference_mode():
        assert model(torch.randn(2, 6, 64, 64)).shape == (2, 11)
        assert model.forward_features(torch.randn(2, 6, 64, 64)).shape == (2, 1024)


def test_even_sampler_has_no_padding_duplicates():
    shards = [list(EvenTrainSampler(23, rank, 4, seed=7)) for rank in range(4)]
    assert len({len(values) for values in shards}) == 1
    flattened = [value for shard in shards for value in shard]
    assert len(flattened) == 20
    assert len(set(flattened)) == 20


def test_cutmix_preserves_tensor_and_label_contract():
    torch.manual_seed(2); np.random.seed(2)
    images = torch.arange(4 * 6 * 16 * 16, dtype=torch.float32).reshape(4, 6, 16, 16)
    labels = torch.arange(4)
    mixed, labels_a, labels_b, lam = cutmix(images, labels, 1.0)
    assert mixed.shape == images.shape
    assert torch.equal(labels_a, labels)
    assert sorted(labels_b.tolist()) == labels.tolist()
    assert 0.0 <= lam <= 1.0


def test_huvec_crossfits_cover_every_target_once(tmp_path: Path):
    rows = []
    global_index = 0
    inventory = [("HUVEC", f"HUVEC-{i:02d}", "train" if i < 16 else "test")
                 for i in range(24)]
    inventory += [("RPE", f"RPE-{i:02d}", "train") for i in range(17)]
    for experiment, (cell, name, dataset) in enumerate(inventory):
        for site in (1, 2):
            rows.append({
                "global_index": global_index, "well_id": f"{name}_A01",
                "cell_type": cell, "dataset": dataset, "experiment": experiment,
                "experiment_name": name, "label": 0, "site": site,
                "relative_path": f"images/{name}/Plate1/A01_s{site}.png",
                "well_type": "treatment",
            })
            global_index += 1
    manifest = tmp_path / "all_sites.parquet"
    pd.DataFrame(rows).to_parquet(manifest, index=False)
    registry = freeze(manifest, tmp_path / "splits")
    target_names = [name for fold in registry["folds"]
                    for name in fold["target_experiment_names"]]
    assert len(target_names) == 24
    assert len(set(target_names)) == 24
    for fold in registry["folds"]:
        assert len(fold["target_experiment_names"]) == 4
        assert len(fold["selection_validation_experiment_names"]) == 4
        assert [row["n_huvec_training_experiments"] for row in fold["scales"]] == [4, 8, 12, 16]
