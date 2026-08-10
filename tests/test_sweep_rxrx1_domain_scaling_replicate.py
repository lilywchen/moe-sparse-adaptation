import copy
import json

import pytest

from scripts.aggregate_rxrx1_domain_scaling_replicate import (
    normalized_config,
    render_report,
)
from scripts.sweep_rxrx1_domain_scaling_replicate import (
    ARMS,
    SEEDS,
    sharded_rows,
    validate_planned_pairs,
    wave_rows,
    write_manifest,
)


def test_registry_is_two_seed_four_arm_quarter_replication():
    rows = wave_rows()
    assert len(rows) == 8
    assert {row[2] for row in rows} == set(SEEDS)
    assert {row[1] for row in rows} == {arm for arm, _ in ARMS}
    assert len({row[5] for row in rows}) == 8
    for _label, _arm, _seed, environments, overrides, _run_id, cfg in rows:
        assert len(environments) == 8
        assert cfg["train"]["environment_subset"] == list(environments)
        assert "analysis.run_mechanism=true" in overrides
        assert "stage=3" in overrides
        assert "train.save_checkpoint_epochs=[30]" in overrides


def test_planned_configs_match_completed_full_anchor_protocol():
    rows = wave_rows()
    assert validate_planned_pairs(rows)


def test_four_container_shards_are_disjoint_and_two_runs_each():
    rows = wave_rows()
    shards = [sharded_rows(rows, index, 4) for index in range(4)]
    assert [len(shard) for shard in shards] == [2, 2, 2, 2]
    assert {row[5] for shard in shards for row in shard} == {row[5] for row in rows}


def test_normalized_config_allows_only_endpoint_and_tag_changes():
    cfg = wave_rows()[0][6]
    changed = copy.deepcopy(cfg)
    changed["run_tag"] = "different"
    changed["train"].pop("environment_subset")
    changed["model"].pop("router_frozen")
    assert normalized_config(cfg) == normalized_config(changed)
    changed["model"]["router_frozen"] = True
    assert normalized_config(cfg) != normalized_config(changed)
    changed["model"]["router_frozen"] = False
    changed["train"]["optim"]["lr"] *= 2
    assert normalized_config(cfg) != normalized_config(changed)


def _result(row, environments):
    return {
        "run_id": row[5], "seed": row[2], "stage": 3, "test_evaluated": True,
        "selection_split": "ood_val", "git_dirty": False, "acc_train": 1.0,
        "acc_within": 0.1, "acc_val": 0.05, "acc_heldout": 0.06,
        "worst_env_heldout": 0.02, "config": row[6],
    }


def test_manifest_and_status_join_frozen_full_anchors(tmp_path, monkeypatch):
    rows = wave_rows()
    quarter = tmp_path / "quarter"
    full = tmp_path / "full"
    monkeypatch.setattr(
        "scripts.sweep_rxrx1_domain_scaling_replicate._dataset_audit",
        lambda _config: {"quarter": {"n_classes_observed": 1139}})
    manifest = write_manifest(quarter, rows)
    manifest["full_anchor_root"] = str(full)
    (quarter / "wave_manifest.json").write_text(json.dumps(manifest))

    full_runs = []
    for row in rows:
        result = _result(row, row[3])
        (quarter / f"{row[5]}.json").write_text(json.dumps(result))
        full_cfg = copy.deepcopy(row[6])
        full_cfg["train"].pop("environment_subset")
        full_run = "full-" + row[5]
        full_runs.append({"label": "full_" + row[0], "arm": row[1], "seed": row[2],
                          "run_id": full_run})
        result["run_id"] = full_run
        result["config"] = full_cfg
        result["acc_heldout"] = 0.10
        (full / f"{full_run}.json").parent.mkdir(parents=True, exist_ok=True)
        (full / f"{full_run}.json").write_text(json.dumps(result))
    (full / "wave_manifest.json").write_text(json.dumps({
        "campaign": "full", "expected_runs": 8, "runs": full_runs,
    }))
    report = render_report(quarter, full)
    assert "8/8 new rows complete" in report
    assert "Architecture-by-scale interactions" in report
    assert "+0.000" in report
    assert "Protocol: all available artifacts and paired configs pass" in report


def test_planned_pairing_rejects_optimizer_drift(monkeypatch):
    rows = wave_rows()
    bad = list(rows)
    cfg = copy.deepcopy(bad[0][6])
    cfg["train"]["optim"]["lr"] *= 2
    bad[0] = (*bad[0][:-1], cfg)
    with pytest.raises(ValueError, match="config drift"):
        validate_planned_pairs(bad)
