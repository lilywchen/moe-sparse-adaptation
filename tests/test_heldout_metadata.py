import pytest

from scripts.run_ccas import normalize_withheld_ood_fields, validate_stage1_artifacts


def _selection_result():
    return {
        "run_id": "run",
        "selection_split": "ood_val",
        "test_evaluated": False,
        "acc_heldout": None,
        "worst_env_heldout": None,
        "per_env_heldout": None,
        "per_env_n_heldout": None,
        "degradation_gap_test": None,
        "acc_selection": 0.2,
        "acc_val": 0.2,
        "worst_env_val": 0.1,
        "acc_within": 0.3,
    }


def test_publish_validation_requires_null_per_environment_test_fields():
    result = _selection_result()
    validate_stage1_artifacts(result)
    result["per_env_heldout"] = {}
    with pytest.raises(ValueError, match="remain null"):
        validate_stage1_artifacts(result)


def test_normalize_withheld_ood_fields_only_repairs_empty_legacy_maps():
    result = _selection_result()
    result["per_env_heldout"] = {}
    result["per_env_n_heldout"] = {}
    normalized = normalize_withheld_ood_fields(result)
    assert normalized["per_env_heldout"] is None
    assert normalized["per_env_n_heldout"] is None
    assert normalized["acc_selection"] == result["acc_selection"]
    assert result["per_env_heldout"] == {}

    result["per_env_heldout"] = {"7": 0.1}
    with pytest.raises(ValueError, match="non-empty OOD-test field"):
        normalize_withheld_ood_fields(result)
