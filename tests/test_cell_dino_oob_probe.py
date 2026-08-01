import importlib.util
from pathlib import Path

import torch


SPEC = importlib.util.spec_from_file_location(
    "probe_rxrx1_cell_dino_oob",
    Path(__file__).resolve().parents[1] / "scripts" / "probe_rxrx1_cell_dino_oob.py",
)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def test_frozen_readouts_separate_classes_and_environments():
    train_f = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    train_f = torch.nn.functional.normalize(train_f, dim=1)
    train_y = torch.tensor([0, 0, 1, 1])
    query_f = torch.tensor([[0.95, 0.05], [0.05, 0.95]])
    query_f = torch.nn.functional.normalize(query_f, dim=1)
    query_y = torch.tensor([0, 1])
    query_env = torch.tensor([7, 9])

    got = PROBE.evaluate_readouts(
        train_f, train_y, query_f, query_y, query_env, n_classes=2,
        device=torch.device("cpu"), query_chunk=1,
    )

    assert got["cosine_1nn"]["accuracy"] == 1.0
    assert got["nearest_centroid"]["accuracy"] == 1.0
    assert got["cosine_1nn"]["worst_env_accuracy"] == 1.0
    assert got["nearest_centroid"]["per_env_n"] == {"7": 1, "9": 1}


def test_centroids_reject_missing_training_classes():
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 1])

    try:
        PROBE.class_centroids(features, labels, n_classes=3)
    except RuntimeError as exc:
        assert "missing 1 classes" in str(exc)
    else:
        raise AssertionError("missing classes must fail closed")
