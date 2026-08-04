import pytest

from scripts.finalize_ccas_checkpoint import validate_recovery_checkpoint


def checkpoint_payload():
    cfg = {
        "dataset": "rxrx1",
        "seed": 0,
        "stage": 1,
        "model": {
            "variant": "moe",
            "placement": "middle",
            "ffn_block_indices": [10, 11],
            "routing_unit": "token",
            "geometry": "cosine",
            "balance": "global",
            "n_experts": 8,
            "top_k": 1,
        },
        "train": {"epochs": 5},
    }
    from moe_shift.capacity.naming import run_id_from

    rid = run_id_from(cfg)
    return {
        "run_id": rid,
        "epoch": 5,
        "config": cfg,
        "model": {},
        "milestone": {
            "run_id": rid,
            "epoch": 5,
            "acc_train": 0.1,
            "acc_within": 0.08,
            "acc_selection": 0.04,
            "worst_env_val": 0.01,
            "selection_split": "ood_val",
            "test_evaluated": False,
        },
    }


def test_recovery_checkpoint_accepts_complete_terminal_selection_checkpoint():
    payload = checkpoint_payload()
    cfg, rid, epoch, milestone = validate_recovery_checkpoint(payload, payload["run_id"])
    assert cfg is payload["config"]
    assert rid == payload["run_id"]
    assert epoch == 5
    assert milestone["test_evaluated"] is False


@pytest.mark.parametrize("mutation", ["partial", "stage3", "wrong_split", "wrong_run"])
def test_recovery_checkpoint_fails_closed(mutation):
    payload = checkpoint_payload()
    if mutation == "partial":
        payload["epoch"] = 2
        payload["milestone"]["epoch"] = 2
    elif mutation == "stage3":
        payload["config"]["stage"] = 3
    elif mutation == "wrong_split":
        payload["milestone"]["selection_split"] = "ood_test"
    else:
        payload["run_id"] = "wrong"
        payload["milestone"]["run_id"] = "wrong"
    with pytest.raises(ValueError):
        validate_recovery_checkpoint(payload)
