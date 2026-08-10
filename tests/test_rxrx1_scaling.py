import torch

from moe_shift.data.rxrx1_scaling import (
    TRAIN_EXPERIMENTS_BY_CELL,
    audit_environment_subset,
    full_environment_subset,
    midpoint_environment_subset,
    quarter_environment_subset,
    subset_digest,
)


def test_quarter_prefix_is_deterministic_balanced_and_nested():
    quarter = quarter_environment_subset()
    midpoint = midpoint_environment_subset()
    full = full_environment_subset()
    assert quarter == (5, 2, 14, 12, 17, 15, 41, 46)
    assert midpoint == (5, 2, 4, 14, 12, 17, 15, 13, 21, 20, 22, 41, 35, 37, 46, 47)
    assert len(quarter) == 8
    assert len(midpoint) == 16
    assert len(full) == 33
    assert set(quarter) < set(midpoint) < set(full)
    cell_by_environment = {
        environment: cell
        for cell, environments in TRAIN_EXPERIMENTS_BY_CELL.items()
        for environment in environments
    }
    assert {cell_by_environment[value] for value in quarter} == {0, 1, 2, 3}
    assert {cell_by_environment[value] for value in midpoint} == {0, 1, 2, 3}
    assert subset_digest(quarter) == subset_digest(reversed(quarter))


def test_audit_counts_fields_wells_sites_classes_and_cells():
    class FakeSubset:
        indices = torch.arange(8)

    class FakeDataset:
        metadata_fields = ["cell_type", "experiment", "plate", "well", "site"]
        metadata_array = torch.tensor([
            [0, 5, 0, 0, 0], [0, 5, 0, 0, 1],
            [1, 14, 1, 0, 0], [1, 14, 1, 0, 1],
            [2, 41, 2, 0, 0], [2, 41, 2, 0, 1],
            [3, 46, 3, 0, 0], [3, 46, 3, 0, 1],
        ])
        y_array = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        n_classes = 2

        @staticmethod
        def get_subset(name):
            assert name == "train"
            return FakeSubset()

    audit = audit_environment_subset(FakeDataset(), [5, 14, 41, 46])
    assert audit["n_environments"] == 4
    assert audit["n_fields"] == 8
    assert audit["n_wells"] == 4
    assert audit["n_sites"] == 8
    assert audit["n_classes_observed"] == 2
    assert audit["examples_per_class_min"] == 4
    assert audit["cell_environment_counts"] == {0: 1, 1: 1, 2: 1, 3: 1}


def test_environment_subset_identity_depends_on_exact_membership():
    from moe_shift.capacity.naming import run_id_from
    from moe_shift.utils.config import apply_overrides, load_config

    left = apply_overrides(load_config("configs/ccas_rxrx1_cell_dino_native.yaml"),
                           ["train.environment_subset=[1,2,3]"])
    right = apply_overrides(load_config("configs/ccas_rxrx1_cell_dino_native.yaml"),
                            ["train.environment_subset=[1,2,4]"])
    assert run_id_from(left) != run_id_from(right)
    assert "envsub3-" in run_id_from(left)
