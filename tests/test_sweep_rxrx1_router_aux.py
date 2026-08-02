import subprocess

from scripts.sweep_rxrx1_router_aux import (
    AUX_SETTINGS,
    PRESSURES,
    _busy_gpu_indices,
    active_marker_live,
    cells,
    clear_active_marker,
    idle_nvidia_slots,
    sharded_rows,
)


def test_router_aux60_has_complete_unique_budget():
    rows = cells()
    assert len(rows) == len(PRESSURES) * len(AUX_SETTINGS) == 16
    assert len({run_id for _, _, run_id in rows}) == 16
    assert sum(tag.startswith("route_") for tag, _, _ in rows) == 8
    assert sum(tag.startswith("canonical_") for tag, _, _ in rows) == 8


def test_router_aux60_changes_only_predeclared_auxiliary_mechanism_fields():
    for tag, overrides, run_id in cells():
        pressure = tag.split("_", 1)[0]
        assert "seed=0" in overrides
        assert "stage=1" in overrides
        assert "model.variant=moe" in overrides
        assert "model.placement=early" in overrides
        assert "model.routing_unit=token" in overrides
        assert "model.geometry=cosine" in overrides
        assert f"model.pressure={pressure}" in overrides
        assert "model.n_experts=8" in overrides
        assert "model.top_k=1" in overrides
        assert "train.epochs=60" in overrides
        assert "train.milestone_epochs=[10,30,60]" in overrides
        assert "train.save_checkpoint_epochs=[10,30,60]" in overrides
        assert "analysis.run_mechanism=true" in overrides
        assert any(v.startswith("losses.balance_w=") for v in overrides)
        assert any(v.startswith("losses.zloss_w=") for v in overrides)
        assert "router_aux60" in run_id


def test_router_aux60_does_not_duplicate_frozen_factorial_default():
    assert ("bw1em2_z1em3", 1.0e-2, 1.0e-3) not in AUX_SETTINGS


def test_router_aux60_shards_are_disjoint_and_exhaustive():
    rows = cells()
    shards = [sharded_rows(rows, index, 5) for index in range(5)]
    flattened = [run_id for shard in shards for _, _, run_id in shard]
    assert len(flattened) == len(rows)
    assert set(flattened) == {run_id for _, _, run_id in rows}
    assert max(map(len, shards)) - min(map(len, shards)) <= 1


def test_router_aux60_maps_physical_processes_to_local_gpu_indices():
    gpu_rows = "0, GPU-a\n1, GPU-b\n"
    process_rows = "GPU-b, 202\nGPU-b, 303\n"
    assert _busy_gpu_indices(gpu_rows, process_rows) == {"1"}


def test_router_aux60_refill_uses_only_physically_idle_slots(monkeypatch):
    outputs = iter(["0, GPU-a\n1, GPU-b\n", "GPU-a, 101\n"])

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout=next(outputs))

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert idle_nvidia_slots(["0", "1"]) == ["1"]


def test_router_aux60_refill_fails_closed_when_occupancy_is_unknown(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert idle_nvidia_slots(["0", "1"]) == []


def test_router_aux60_restart_marker_preserves_live_worker(tmp_path, monkeypatch):
    marker = tmp_path / "run.active"
    marker.write_text("123")
    monkeypatch.setattr("scripts.sweep_rxrx1_router_aux.os.kill", lambda pid, signal: None)
    assert active_marker_live(marker)
    assert marker.read_text() == "123"
    clear_active_marker(marker, 999)
    assert marker.is_file()
    clear_active_marker(marker, 123)
    assert not marker.exists()


def test_router_aux60_restart_marker_removes_dead_worker(tmp_path, monkeypatch):
    marker = tmp_path / "run.active"
    marker.write_text("123")

    def missing(pid, signal):
        raise ProcessLookupError(pid)

    monkeypatch.setattr("scripts.sweep_rxrx1_router_aux.os.kill", missing)
    assert not active_marker_live(marker)
    assert not marker.exists()
