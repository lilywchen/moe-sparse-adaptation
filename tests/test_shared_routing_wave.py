import json

from scripts.aggregate_shared_routing import render_report
from scripts.sweep_rxrx1_shared_routing import sharded_rows, wave_rows, write_manifest


def test_registry_is_four_routing_controls_at_two_anchor_seeds():
    rows = wave_rows()
    assert len(rows) == 8
    assert {row[2] for row in rows} == {1, 2}
    assert {row[1] for row in rows} == {
        "shared_E3k1_image", "shared_E3k1_balance1e3",
        "shared_E3k1_balance0", "shared_E3k1_router_frozen",
    }
    assert len({row[4] for row in rows}) == 8
    for _display, _arm, _seed, overrides, _run_id, cfg in rows:
        assert cfg["model"]["variant"] == "shared_moe"
        assert cfg["model"]["n_experts"] == 3
        assert cfg["model"]["top_k"] == 1
        assert cfg["model"]["ffn_block_indices"] == [10, 11]
        assert "stage=3" in overrides
        assert "train.save_checkpoint_epochs=[30]" in overrides


def test_controls_change_only_predeclared_routing_factors():
    by_arm = {row[1]: row[5] for row in wave_rows() if row[2] == 1}
    assert by_arm["shared_E3k1_image"]["model"]["routing_unit"] == "image"
    assert by_arm["shared_E3k1_balance1e3"]["losses"]["balance_w"] == 1e-3
    assert by_arm["shared_E3k1_balance0"]["losses"]["balance_w"] == 0.0
    assert by_arm["shared_E3k1_router_frozen"]["model"]["router_frozen"] is True


def test_two_container_shards_are_disjoint_and_four_runs_each():
    rows = wave_rows()
    shards = [sharded_rows(rows, index, 2) for index in range(2)]
    assert [len(shard) for shard in shards] == [4, 4]
    assert {row[4] for shard in shards for row in shard} == {row[4] for row in rows}


def test_manifest_and_status_are_one_command_ready(tmp_path, monkeypatch):
    rows = wave_rows()
    monkeypatch.setattr("scripts.sweep_rxrx1_shared_routing.ANCHOR_ROOT", tmp_path / "anchors")
    write_manifest(tmp_path, rows)
    anchor_root = tmp_path / "anchors"
    anchor_root.mkdir()
    anchor_runs = []
    for seed in (1, 2):
        run_id = f"anchor_s{seed}"
        anchor_runs.append({"label": run_id, "arm": "shared_E3k1_late2", "seed": seed,
                            "run_id": run_id})
        (anchor_root / f"{run_id}.json").write_text(json.dumps({
            "seed": seed, "selection_split": "ood_val", "stage": 3,
            "test_evaluated": True, "config": {"train": {"epochs": 30}},
            "acc_val": 0.22,
        }))
    (anchor_root / "wave_manifest.json").write_text(json.dumps({"runs": anchor_runs}))
    first = rows[0]
    (tmp_path / f"{first[4]}.trainlog.jsonl").write_text(json.dumps({"epoch": 9}) + "\n")
    report = render_report(tmp_path)
    assert "8" in report and first[0] in report and "training" in report
    assert "shared_E3k1_late2" in report
