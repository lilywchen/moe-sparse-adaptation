import json

from scripts.aggregate_shared_confirm import render_report
from scripts.sweep_rxrx1_shared_confirm import sharded_rows, wave_rows, write_manifest


def _override(overrides, key):
    prefix = key + "="
    return [value[len(prefix):] for value in overrides if value.startswith(prefix)][-1]


def test_registry_is_four_matched_arms_at_two_fresh_seeds():
    rows = wave_rows()
    assert len(rows) == 8
    assert {row[2] for row in rows} == {1, 2}
    assert {row[1] for row in rows} == {
        "original", "dense_E4_late2", "replace_E4k2_late2", "shared_E3k1_late2",
    }
    assert len({row[4] for row in rows}) == 8
    for _display, arm, _seed, overrides, _run_id, cfg in rows:
        assert "stage=3" in overrides
        assert "analysis.run_mechanism=true" in overrides
        assert "train.label_smoothing=0.0" in overrides
        assert "train.save_checkpoint_epochs=[30]" in overrides
        if arm != "original":
            assert cfg["model"]["ffn_block_indices"] == [10, 11]


def test_capacity_and_active_path_matching_is_explicit():
    by_arm = {row[1]: row for row in wave_rows() if row[2] == 1}
    assert _override(by_arm["dense_E4_late2"][3], "model.n_experts") == "4"
    assert _override(by_arm["replace_E4k2_late2"][3], "model.top_k") == "2"
    assert _override(by_arm["shared_E3k1_late2"][3], "model.n_experts") == "3"
    assert _override(by_arm["shared_E3k1_late2"][3], "model.top_k") == "1"


def test_two_container_shards_are_disjoint_and_four_runs_each():
    rows = wave_rows()
    shards = [sharded_rows(rows, index, 2) for index in range(2)]
    assert [len(shard) for shard in shards] == [4, 4]
    assert {row[4] for shard in shards for row in shard} == {row[4] for row in rows}


def test_manifest_and_status_are_single_command_ready(tmp_path):
    rows = wave_rows()
    write_manifest(tmp_path, rows)
    first = rows[0]
    (tmp_path / f"{first[4]}.milestones.jsonl").write_text(json.dumps({
        "epoch": 10, "acc_selection": 0.2, "acc_within": 0.5, "acc_train": 0.9,
    }) + "\n")
    (tmp_path / f"{first[4]}.trainlog.jsonl").write_text(json.dumps({"epoch": 9}) + "\n")
    report = render_report(tmp_path)
    assert "8" in report
    assert first[0] in report
    assert "20.000%" in report
    assert "training" in report
