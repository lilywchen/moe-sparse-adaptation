import scripts.sweep_rxrx3_core_plate_scaling as sweep


def test_each_plate_point_is_two_seed_four_arm_and_collision_free():
    all_ids = set()
    for plates in (1, 2, 4):
        rows = sweep.wave_rows(plates)
        assert len(rows) == 8
        assert {row[1] for row in rows} == {name for name, _ in sweep.pilot.ARMS}
        assert {row[2] for row in rows} == set(sweep.pilot.SEEDS)
        assert [len(sweep.pilot.sharded_rows(rows, shard, 4)) for shard in range(4)] == [2] * 4
        assert all(str(sweep.manifest_path(plates)) == row[5]["rxrx3_manifest"] for row in rows)
        ids = {row[4] for row in rows}
        assert len(ids) == 8
        assert not all_ids & ids
        all_ids |= ids


def test_manifest_freezes_axis_anchor_and_protocol(tmp_path, monkeypatch):
    rows = sweep.wave_rows(1)
    monkeypatch.setattr(sweep, "_source_identity", lambda: ("abc1234full", False))
    fake_audit = {
        "passed": True, "axis": "train_plate_count_with_four_guides_fixed",
        "selected_plates": 1,
        "split_counts": {"train": 2696, "id_val": 2708, "ood_test": 23855},
    }
    fake_capacity = {
        arm: {"total_params": 100 + index, "active_ffn_params": 10 + index}
        for index, (arm, _) in enumerate(sweep.pilot.ARMS)
    }
    manifest = sweep.write_manifest(
        tmp_path, rows, 1, audit=fake_audit, capacity=fake_capacity
    )
    assert manifest["expected_runs"] == 8
    assert manifest["plate_count"] == 1
    assert manifest["guide_count_fixed"] == 4
    assert manifest["class_count_fixed"] == 674
    assert manifest["train_experiments_fixed"] == 85
    assert manifest["full_anchor_root"] == str(sweep.FULL_ANCHOR_ROOT)
    assert manifest["source_git_commit"] == "abc1234full"
    assert all(spec["resolved_config"]["rxrx3_manifest"].endswith(
        "rxrx3_core_gene_plate_1.tsv") for spec in manifest["runs"])


def test_plate_manifest_names_and_expected_counts_are_frozen():
    assert sweep.EXPECTED_TRAIN_ROWS == {1: 2696, 2: 5376, 4: 10706, 8: 21404}
    assert sweep.manifest_path(4).name == "rxrx3_core_gene_plate_4.tsv"
    try:
        sweep.manifest_path(3)
    except ValueError as error:
        assert "plates must be one of" in str(error)
    else:
        raise AssertionError("unsupported plate count was accepted")
