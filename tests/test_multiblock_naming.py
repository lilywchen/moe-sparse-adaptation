from moe_shift.capacity.naming import run_id_from


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
