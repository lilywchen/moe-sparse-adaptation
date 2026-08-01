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
