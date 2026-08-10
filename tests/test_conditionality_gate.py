import json

import pytest

from scripts.aggregate_rxrx1_conditionality_gate import _paired_domain_bootstrap, render_report
from scripts import sweep_rxrx1_conditionality_gate as sweep


def test_conditionality_gate_is_three_paired_quartets_with_unique_ids():
    rows = sweep.wave_rows()
    assert len(rows) == 12
    assert len({row[4] for row in rows}) == 12
    assert [len(sweep.sharded_rows(rows, index, 4)) for index in range(4)] == [3, 3, 3, 3]
    for seed in sweep.SEEDS:
        quartet = [row for row in rows if row[2] == seed]
        assert {row[1] for row in quartet} == {arm for arm, _ in sweep.ARMS}
        assert {row[5]["train"]["data_seed"] for row in quartet} == {seed}
        assert {row[5]["train"]["model_seed"] for row in quartet} == {seed}
        assert {row[5]["train"]["training_seed"] for row in quartet} == {seed}
        assert {row[5]["train"]["label_smoothing"] for row in quartet} == {0.1}
    full_st = [row for row in rows if row[1] == "shared_E3_fullST"]
    assert all("full_st" in row[4] for row in full_st)


def test_config_gate_rejects_nonarchitecture_drift():
    rows = sweep.wave_rows()
    rows[1][5]["train"]["batch_size"] = 32
    with pytest.raises(ValueError, match="batch_size"):
        sweep.validate_resolved_configs(rows)


def test_manifest_records_stop_rule_and_separate_training_compute(tmp_path, monkeypatch):
    rows = sweep.wave_rows()
    monkeypatch.setattr(sweep, "_source_identity", lambda: ("a" * 40, False))
    capacity = {
        arm: {"active_ffn_params": 100, "inference_active_ffn_params": 100,
              "training_active_converted_ffn_params": 200 if arm == "shared_E3_fullST" else 100}
        for arm, _ in sweep.ARMS
    }
    payload = sweep.write_manifest(
        tmp_path, rows, audit={"passed": True}, capacity=capacity)
    assert payload["expected_runs"] == 12
    assert "Only generic-router salvage wave" in payload["stopping_rule"]
    assert payload["compute_accounting"]["shared_E3_fullST"][
        "training_active_converted_ffn_params"] == 200


def _row(seed, deltas):
    names = [str(index) for index in range(len(deltas))]
    return {"seed": seed, "result": {
        "per_env_heldout": dict(zip(names, deltas)),
        "per_env_n_heldout": {name: 10 for name in names},
    }}


def test_hierarchical_bootstrap_uses_paired_domain_differences():
    left = [_row(1, [0.3, 0.4]), _row(2, [0.5, 0.6])]
    right = [_row(1, [0.2, 0.3]), _row(2, [0.4, 0.5])]
    summary = _paired_domain_bootstrap(left, right, draws=1000, seed=7)
    assert summary["estimate"] == pytest.approx(0.1)
    assert summary["ci95"][0] == pytest.approx(0.1)
    assert summary["ci95"][1] == pytest.approx(0.1)


def test_status_is_one_command_on_pending_manifest(tmp_path, monkeypatch):
    rows = sweep.wave_rows()
    monkeypatch.setattr(sweep, "_source_identity", lambda: ("b" * 40, False))
    capacity = {arm: {} for arm, _ in sweep.ARMS}
    sweep.write_manifest(tmp_path, rows, audit={"passed": True}, capacity=capacity)
    report = render_report(tmp_path)
    assert "0/12 complete" in report
    assert "pending" in report
