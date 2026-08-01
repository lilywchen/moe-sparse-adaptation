from pathlib import Path

from scripts.pilot_rxrx1 import _rows


ROOT = Path(__file__).resolve().parents[1]


def test_native_instrument_manifests_have_distinct_run_id_namespaces():
    cell = _rows(
        "instrument", config=str(ROOT / "configs/ccas_rxrx1_cell_dino_native.yaml"))
    adaptive = _rows(
        "instrument", config=str(ROOT / "configs/ccas_rxrx1_channel_adaptive_dino.yaml"))

    assert len(cell) == len(adaptive) == 2
    assert all("cell_dino_cell_dino_native_cp5" in rid for _, _, rid in cell)
    assert all("channel_adaptive_dino_native6" in rid for _, _, rid in adaptive)
    assert set(rid for _, _, rid in cell).isdisjoint(rid for _, _, rid in adaptive)


def test_null_gate_diagnosis_is_only_frozen_and_random_route_pair():
    rows = _rows(
        "diagnose", config=str(ROOT / "configs/ccas_rxrx1_cell_dino_native.yaml"),
        lr=1.0e-4, epochs=10)

    assert [tag for tag, _, _ in rows] == [
        "diag_learned_random_route", "diag_frozen_router"]
    assert all("s0" in rid for _, _, rid in rows)
    assert "moe_middle_token_cosine_canonical_E8k1" in rows[0][2]
    assert "moe_frozen_middle_token_cosine_canonical_E8k1" in rows[1][2]
    for _, overrides, _ in rows:
        assert "stage=2" in overrides
        assert "analysis.run_mechanism=true" in overrides
        assert "train.optim.lr=0.0001" in overrides
        assert "train.epochs=10" in overrides
