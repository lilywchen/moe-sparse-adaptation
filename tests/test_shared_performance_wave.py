import json

import torch

from moe_shift.data.rxrx1 import CrossExperimentBatchSampler
from scripts.summarize_rxrx1_performance_wave import render_table
from scripts.sweep_rxrx1_shared_performance import sharded_rows, wave_rows, write_manifest


def _override(overrides, key):
    prefix = key + "="
    return [value[len(prefix):] for value in overrides if value.startswith(prefix)][-1]


def test_wave_is_eight_unique_stage3_runs_with_two_bounded_robustness_arms():
    rows = wave_rows()
    assert len(rows) == 8
    assert len({run_id for _, _, run_id, _ in rows}) == 8
    by_label = {label: overrides for label, overrides, _run_id, _cfg in rows}
    assert set(by_label) == {
        "replace_E4k2_late2", "shared_E3k1_late2", "shared_E3k2_late2",
        "shared_E7k1_late2", "shared_E3k1_late4", "shared_E3k2_late4",
        "shared_E3k1_xbatch", "shared_E3k1_mixstyle",
    }
    for overrides in by_label.values():
        assert "stage=3" in overrides
        assert "train.save_checkpoint_epochs=[30]" in overrides
        assert "analysis.run_mechanism=false" in overrides
    assert _override(by_label["shared_E3k1_xbatch"], "train.cross_experiment_pairs") == "true"
    assert _override(
        by_label["shared_E3k1_xbatch"], "losses.cross_experiment_contrastive_w") == "0.1"
    assert _override(
        by_label["shared_E3k1_mixstyle"], "model.feature_stat_mix_prob") == "0.5"


def test_four_container_shards_are_disjoint_and_two_runs_each():
    rows = wave_rows()
    shards = [sharded_rows(rows, index, 4) for index in range(4)]
    assert [len(shard) for shard in shards] == [2, 2, 2, 2]
    assert {run_id for shard in shards for _, _, run_id, _ in shard} == {
        run_id for _, _, run_id, _ in rows}


def test_cross_experiment_sampler_forms_label_pairs_across_experiments_within_cell_type():
    labels, experiments, cells = [], [], []
    for cell in (0, 1):
        for label in range(8):
            for experiment in range(3):
                labels.append(label); experiments.append(experiment + 10 * cell); cells.append(cell)
    sampler = CrossExperimentBatchSampler(labels, experiments, cells, batch_size=8)
    batch = next(iter(sampler))
    batch_labels = torch.tensor(labels)[batch]
    batch_experiments = torch.tensor(experiments)[batch]
    batch_cells = torch.tensor(cells)[batch]
    assert batch_cells.unique().numel() == 1
    for label in batch_labels.unique():
        selected = batch_labels == label
        assert int(selected.sum()) == 2
        assert batch_experiments[selected].unique().numel() == 2


def test_manifest_and_live_summary_merge_milestone_and_terminal_results(tmp_path):
    rows = wave_rows()
    write_manifest(tmp_path, rows)
    first_label, _overrides, first_id, _cfg = rows[0]
    (tmp_path / f"{first_id}.milestones.jsonl").write_text(json.dumps({
        "epoch": 10, "acc_selection": 0.22, "acc_within": 0.4, "acc_train": 0.9,
    }) + "\n")
    (tmp_path / f"{first_id}.trainlog.jsonl").write_text(json.dumps({"epoch": 10}) + "\n")
    table = render_table(tmp_path)
    assert first_label in table
    assert "22.000%" in table
    assert "training" in table
