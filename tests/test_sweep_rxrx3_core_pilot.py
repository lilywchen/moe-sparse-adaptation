import json

import scripts.sweep_rxrx3_core_pilot as sweep
from scripts.aggregate_rxrx3_core_pilot import load, render_report, validate


def test_pilot_has_exact_matched_two_seed_four_arm_design():
    rows = sweep.wave_rows()
    assert len(rows) == 8
    assert len({row[4] for row in rows}) == 8
    assert {row[1] for row in rows} == {name for name, _ in sweep.ARMS}
    assert {row[2] for row in rows} == set(sweep.SEEDS)
    assert sweep.validate_resolved_configs(rows)
    assert [len(sweep.sharded_rows(rows, shard, 4)) for shard in range(4)] == [2, 2, 2, 2]


def test_config_gate_rejects_nonarchitecture_drift():
    rows = sweep.wave_rows()
    rows[1][5]["train"]["batch_size"] = 32
    try:
        sweep.validate_resolved_configs(rows)
    except ValueError as error:
        assert "train.batch_size" in str(error)
    else:
        raise AssertionError("unexpected training drift was accepted")


def test_manifest_and_aggregator_validate_terminal_artifacts(tmp_path, monkeypatch):
    rows = sweep.wave_rows()
    monkeypatch.setattr(sweep, "_source_identity", lambda: ("abc1234full", False))
    fake_audit = {"passed": True, "split_counts": {
        "train": 21404, "id_val": 2708, "ood_test": 23855,
    }}
    fake_capacity = {
        arm: {"total_params": 100 + index, "active_ffn_params": 10 + index,
              "estimated_active_ffn_flops_relative": 1.0}
        for index, (arm, _) in enumerate(sweep.ARMS)
    }
    manifest = sweep.write_manifest(
        tmp_path, rows, audit=fake_audit, capacity=fake_capacity
    )
    for spec in manifest["runs"]:
        result = {
            "run_id": spec["run_id"], "dataset": "rxrx3_core", "seed": spec["seed"],
            "stage": 3, "test_evaluated": True, "selection_split": "id_val",
            "git_sha": "abc1234", "git_dirty": False, "config": spec["resolved_config"],
            "acc_train": 0.5, "acc_within": 0.2, "acc_val": 0.2,
            "acc_heldout": 0.1, "worst_env_heldout": 0.01,
            "per_env_heldout": {str(index): 0.1 for index in range(85)},
            "per_env_n_heldout": {str(index): (281 if index < 55 else 280)
                                  for index in range(85)},
            "total_params": fake_capacity[spec["arm"]]["total_params"],
            "active_ffn_params": fake_capacity[spec["arm"]]["active_ffn_params"],
        }
        # 55*281 + 30*280 = 23,855.
        (tmp_path / f"{spec['run_id']}.json").write_text(json.dumps(result))
    _root, loaded_manifest, loaded_rows = load(tmp_path)
    assert validate(loaded_manifest, loaded_rows) == []
    report = render_report(tmp_path)
    assert "8/8 complete" in report
    assert "Worst OOD experiment" in report

