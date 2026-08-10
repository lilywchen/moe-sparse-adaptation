import copy
import json

import pytest

from scripts.aggregate_rxrx1_domain_midpoint import render_report
from scripts.sweep_rxrx1_domain_midpoint import (
    ARMS,
    SEEDS,
    sharded_rows,
    validate_planned_pairs,
    wave_rows,
    write_manifest,
)


def test_registry_is_two_seed_four_arm_midpoint():
    rows = wave_rows()
    assert len(rows) == 8
    assert {row[2] for row in rows} == set(SEEDS)
    assert {row[1] for row in rows} == {arm for arm, _ in ARMS}
    assert len({row[5] for row in rows}) == 8
    for _label, _arm, _seed, environments, overrides, _run_id, cfg in rows:
        assert len(environments) == 16
        assert cfg["train"]["environment_subset"] == list(environments)
        assert "analysis.run_mechanism=true" in overrides
        assert "stage=3" in overrides
        assert "train.save_checkpoint_epochs=[30]" in overrides


def test_midpoint_configs_match_completed_full_anchor_protocol():
    assert validate_planned_pairs(wave_rows())


def test_four_container_shards_are_disjoint_and_two_runs_each():
    rows = wave_rows()
    shards = [sharded_rows(rows, index, 4) for index in range(4)]
    assert [len(shard) for shard in shards] == [2, 2, 2, 2]
    assert {row[5] for shard in shards for row in shard} == {row[5] for row in rows}


def _result(run_id, seed, cfg, value):
    return {
        "run_id": run_id, "seed": seed, "stage": 3, "test_evaluated": True,
        "selection_split": "ood_val", "git_dirty": False, "acc_train": 1.0,
        "acc_within": value, "acc_val": value, "acc_heldout": value,
        "worst_env_heldout": value, "config": cfg,
    }


def test_manifest_and_report_join_three_protocol_matched_points(tmp_path, monkeypatch):
    rows = wave_rows()
    quarter = tmp_path / "quarter"
    midpoint = tmp_path / "midpoint"
    full = tmp_path / "full"
    monkeypatch.setattr(
        "scripts.sweep_rxrx1_domain_midpoint._dataset_audit",
        lambda _config: {"scales": {"quarter": {}, "midpoint": {}, "full": {}}})
    manifest = write_manifest(midpoint, rows)
    manifest["quarter_anchor_root"] = str(quarter)
    manifest["full_anchor_root"] = str(full)
    (midpoint / "wave_manifest.json").write_text(json.dumps(manifest))

    quarter.mkdir()
    full.mkdir()
    qspecs, fspecs = [], []
    for row in rows:
        label, arm, seed, environments, _overrides, run_id, cfg = row
        midpoint_result = _result(run_id, seed, cfg, 0.20)
        (midpoint / f"{run_id}.json").write_text(json.dumps(midpoint_result))

        quarter_cfg = copy.deepcopy(cfg)
        quarter_cfg["train"]["environment_subset"] = manifest["quarter_environment_subset"]
        qrun = "quarter-" + run_id
        qspecs.append({"label": "quarter_" + label, "arm": arm, "seed": seed,
                       "run_id": qrun})
        (quarter / f"{qrun}.json").write_text(json.dumps(
            _result(qrun, seed, quarter_cfg, 0.10)))

        full_cfg = copy.deepcopy(cfg)
        full_cfg["train"].pop("environment_subset")
        frun = "full-" + run_id
        fspecs.append({"label": "full_" + label, "arm": arm, "seed": seed,
                       "run_id": frun})
        (full / f"{frun}.json").write_text(json.dumps(
            _result(frun, seed, full_cfg, 0.30)))

    (quarter / "wave_manifest.json").write_text(json.dumps({
        "campaign": "quarter", "expected_runs": 8, "runs": qspecs,
    }))
    (full / "wave_manifest.json").write_text(json.dumps({
        "campaign": "full", "expected_runs": 8, "runs": fspecs,
    }))
    report = render_report(midpoint, quarter, full)
    assert "8/8 new rows complete" in report
    assert "three-point configs pass" in report
    assert "points/octave" in report


def test_midpoint_pairing_rejects_optimizer_drift():
    rows = wave_rows()
    bad = list(rows)
    cfg = copy.deepcopy(bad[0][6])
    cfg["train"]["optim"]["lr"] *= 2
    bad[0] = (*bad[0][:-1], cfg)
    with pytest.raises(ValueError, match="config drift"):
        validate_planned_pairs(bad)
