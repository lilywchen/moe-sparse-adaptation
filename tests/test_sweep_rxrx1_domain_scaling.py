import json

from scripts import aggregate_rxrx1_domain_scaling as aggregate
from scripts import sweep_rxrx1_domain_scaling as sweep


def test_wave_is_a_unique_matched_two_by_four_factorial():
    rows = sweep.wave_rows()
    assert len(rows) == 8
    assert len({row[5] for row in rows}) == 8
    assert {(row[1], row[2]) for row in rows} == {
        (scale, arm) for scale in ("quarter", "full") for arm in aggregate.ARMS
    }
    quarter = [row for row in rows if row[1] == "quarter"]
    full = [row for row in rows if row[1] == "full"]
    assert all(len(row[3]) == 8 for row in quarter)
    assert all(row[3] is None for row in full)
    assert {row[6]["seed"] for row in rows} == {sweep.SEED}
    assert {row[6]["train"]["epochs"] for row in rows} == {30}


def test_four_shards_assign_one_scale_pair_per_architecture():
    rows = sweep.wave_rows()
    for shard_index in range(4):
        shard = sweep.sharded_rows(rows, shard_index, 4)
        assert len(shard) == 2
        assert {row[1] for row in shard} == {"quarter", "full"}
        assert len({row[2] for row in shard}) == 1


def _result(row, test, worst, within, val):
    return {
        "seed": row["seed"], "stage": 3, "test_evaluated": True,
        "git_dirty": False, "selection_split": "ood_val",
        "config": {"train": {"epochs": 30,
                               "environment_subset": row["environment_subset"]}},
        "acc_train": 1.0, "acc_within": within, "acc_val": val,
        "acc_heldout": test, "worst_env_heldout": worst,
    }


def test_aggregator_reports_test_first_contrasts_and_scale_interaction(tmp_path):
    rows = sweep.wave_rows()
    specs = [{
        "label": row[0], "scale": row[1], "arm": row[2], "seed": sweep.SEED,
        "run_id": row[5], "environment_subset": list(row[3] or ()),
    } for row in rows]
    manifest = {
        "campaign": sweep.CAMPAIGN,
        "dataset_audit": {
            "quarter": {"n_classes_observed": 1139,
                        "cell_environment_counts": {"0": 2, "1": 4, "2": 1, "3": 1}},
            "full": {"n_classes_observed": 1139,
                     "cell_environment_counts": {"0": 7, "1": 16, "2": 7, "3": 3}},
        },
        "runs": specs,
    }
    (tmp_path / "wave_manifest.json").write_text(json.dumps(manifest))
    values = {
        ("quarter", "original"): .20, ("quarter", "dense_E4_late2"): .25,
        ("quarter", "replace_E4k2_late2"): .23,
        ("quarter", "shared_E3k1_late2"): .24,
        ("full", "original"): .35, ("full", "dense_E4_late2"): .38,
        ("full", "replace_E4k2_late2"): .37, ("full", "shared_E3k1_late2"): .39,
    }
    for spec in specs:
        value = values[(spec["scale"], spec["arm"])]
        (tmp_path / f"{spec['run_id']}.json").write_text(json.dumps(
            _result(spec, value, value / 4, value + .1, value / 2)))
    report = aggregate.render_report(tmp_path)
    assert "8/8 complete" in report
    assert "shared_E3k1_late2 − dense_E4_late2 | +1.000" in report
    # Shared trails dense by one point at quarter and leads by one at full: +2 interaction.
    assert "| OOD test | +2.000 |" in report
    assert "Protocol: declared scale/seed/stage checks pass" in report
