import pytest

from moe_shift.capacity.naming import explicit_block_indices, run_id_from


def test_run_id_encodes_explicit_multiblock_location():
    cfg = {
        "dataset": "rxrx1", "seed": 0,
        "model": {
            "variant": "moe", "placement": "middle", "block_indices": [1, 5, 9],
            "routing_unit": "token", "geometry": "cosine", "balance": "global",
            "n_experts": 8, "top_k": 1,
        },
        "train": {"epochs": 10},
    }
    assert "blocks1-5-9" in run_id_from(cfg)


def test_run_id_encodes_ffn_block_indices_alias():
    cfg = {
        "dataset": "rxrx1", "seed": 0,
        "model": {
            "variant": "moe", "placement": "middle", "ffn_block_indices": [10, 11],
            "routing_unit": "token", "geometry": "cosine", "balance": "global",
            "n_experts": 8, "top_k": 1,
        },
        "train": {"epochs": 2},
    }
    assert "blocks10-11" in run_id_from(cfg)


def test_explicit_block_index_aliases_must_agree():
    with pytest.raises(ValueError, match="disagree"):
        explicit_block_indices({"block_indices": [6], "ffn_block_indices": [11]})
