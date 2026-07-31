"""Guards on the analysis, not the model — deliberately torch-free so they run anywhere.

A wrong aggregator is as fatal to the study as a wrong model: it silently mispairs runs, leaks the
test split, or reports an effect the design cannot support. Each test below plants a known truth
in synthetic result JSONs and checks the aggregator recovers exactly that.
"""
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("agg", ROOT / "scripts" / "aggregate_ccas.py")
agg = importlib.util.module_from_spec(_spec)
sys.modules["agg"] = agg
_spec.loader.exec_module(agg)

PLACEMENTS = ["early", "middle", "late"]
UNITS = ["image", "token"]
GEOMS = ["linear", "cosine"]
PRESSURES = ["canonical", "route", "output"]
SEEDS = [0, 1, 2]


def _write(d, **kw):
    r = {"dataset": "rxrx1", "seed": 0, "variant": "moe", "placement": "middle",
         "routing_unit": "token", "geometry": "cosine", "pressure": "canonical",
         "balance": "global",
         "n_experts": 8, "top_k": 1, "stage": 1, "selection_split": "ood_val",
         "test_evaluated": False, "acc_heldout": None, "acc_within": 0.9,
         "total_params": 30_000_000, "router_params": 3073, "config": {"junk": True}}
    r.update(kw)
    r.setdefault("run_id", f"{r['dataset']}_{r['variant']}_{r['placement']}_{r['routing_unit']}_"
                           f"{r['geometry']}_{r['pressure']}_s{r['seed']}")
    (Path(d) / f"{r['run_id']}.json").write_text(json.dumps(r))
    return r


def make_grid(d, token_effect=0.0, cosine_effect=0.0, base=0.30, wide_acc=0.30, noise=0.0):
    """A full Stage-1 grid with a planted main effect and a matched dense-wide control."""
    rng = np.random.default_rng(0)
    for pl, ru, ge, pressure in itertools.product(PLACEMENTS, UNITS, GEOMS, PRESSURES):
        for s in SEEDS:
            acc = (base + (token_effect if ru == "token" else 0.0)
                   + (cosine_effect if ge == "cosine" else 0.0)
                   + (noise * rng.standard_normal() if noise else 0.0))
            _write(d, variant="moe", placement=pl, routing_unit=ru, geometry=ge,
                   pressure=pressure,
                   balance=("within_environment" if pressure == "route" else "global"),
                   seed=s, acc_selection=acc, acc_val=acc)
    for pl, pressure in itertools.product(PLACEMENTS, ["canonical", "output"]):
        for s in SEEDS:
            _write(d, variant="dense_wide", placement=pl, routing_unit="na", geometry="na",
                   pressure=pressure, balance="na", seed=s,
                   acc_selection=wide_acc, acc_val=wide_acc,
                   run_id=f"rxrx1_dense_wide_{pl}_{pressure}_s{s}")
    for s in SEEDS:
        _write(d, variant="original", placement="na", routing_unit="na", geometry="na",
               balance="na", seed=s, acc_selection=0.25, acc_val=0.25,
               run_id=f"rxrx1_original_s{s}")


# ------------------------------------------------------------------ loading
def test_load_ignores_non_result_files(tmp_path):
    _write(tmp_path, acc_selection=0.3)
    (tmp_path / "x.trainlog.jsonl").write_text('{"epoch": 0}\n')
    (tmp_path / "broken.json").write_text("{not json")
    df = agg.load(tmp_path)
    assert len(df) == 1
    assert "config" not in df.columns, "the bulky config must not be flattened into the frame"


def test_legacy_runs_get_a_selection_metric(tmp_path):
    r = {"dataset": "rxrx1", "seed": 0, "variant": "moe", "placement": "middle",
         "routing_unit": "token", "geometry": "cosine", "pressure": "canonical",
         "balance": "global",
         "acc_heldout": 0.31, "run_id": "legacy_run"}
    (tmp_path / "legacy_run.json").write_text(json.dumps(r))
    df = agg.load(tmp_path)
    assert df.acc_selection.iloc[0] == pytest.approx(0.31)
    assert df.selection_split.iloc[0] == "legacy"


# ------------------------------------------------------------------ pairing
def test_conditional_gain_is_paired_by_placement_and_seed(tmp_path):
    # dense-wide differs BY PLACEMENT, so an unpaired mean would smear the three controls together
    for pl, wide in zip(PLACEMENTS, [0.20, 0.30, 0.40]):
        for s in SEEDS:
            _write(tmp_path, variant="dense_wide", placement=pl, seed=s,
                   acc_selection=wide, run_id=f"w_{pl}_{s}")
            _write(tmp_path, variant="moe", placement=pl, seed=s,
                   acc_selection=wide + 0.05, run_id=f"m_{pl}_{s}")
    moe, notes = agg.paired_contrasts(agg.load(tmp_path))
    assert moe.conditional_gain.round(6).eq(0.05).all(), "pairing must be per placement AND seed"
    assert not notes or all("no dense_wide" not in n for n in notes)


def test_unmatched_controls_are_reported_not_silently_dropped(tmp_path):
    _write(tmp_path, variant="moe", seed=0, acc_selection=0.3, run_id="m0")
    _write(tmp_path, variant="moe", seed=1, acc_selection=0.3, run_id="m1")
    _write(tmp_path, variant="dense_wide", seed=0, acc_selection=0.2, run_id="w0")
    moe, notes = agg.paired_contrasts(agg.load(tmp_path))
    assert moe.conditional_gain.isna().sum() == 1
    assert any("no depth-matched dense_wide" in n for n in notes)


def test_missing_dense_wide_is_a_loud_note(tmp_path):
    _write(tmp_path, variant="moe", seed=0, acc_selection=0.3, run_id="m0")
    _, notes = agg.paired_contrasts(agg.load(tmp_path))
    assert any("PRIMARY" in n for n in notes)


# ------------------------------------------------------------------ statistics
def test_factorial_fit_recovers_a_planted_main_effect(tmp_path):
    make_grid(tmp_path, token_effect=0.06, wide_acc=0.30)
    moe, _ = agg.paired_contrasts(agg.load(tmp_path))
    fit = agg.factorial_fit(moe, "conditional_gain")
    est = dict(zip(fit.term, fit.estimate))
    # +0.06 for token, 0 for image -> deviations from the grand mean are +/- 0.03
    assert est["routing_unit[token]"] == pytest.approx(0.03, abs=1e-6)
    assert est["routing_unit[image]"] == pytest.approx(-0.03, abs=1e-6)
    assert abs(est["geometry[cosine]"]) < 1e-6
    for pl in PLACEMENTS:
        assert abs(est[f"placement[{pl}]"]) < 1e-6


def test_factorial_fit_main_effects_sum_to_zero(tmp_path):
    make_grid(tmp_path, token_effect=0.04, cosine_effect=-0.02, noise=0.01)
    moe, _ = agg.paired_contrasts(agg.load(tmp_path))
    fit = agg.factorial_fit(moe, "conditional_gain")
    main = fit[~fit.term.str.contains(":") & (fit.term != "intercept")]
    for f in agg.FACTORS:
        rows = main[main.term.str.startswith(f + "[")]
        assert rows.estimate.sum() == pytest.approx(0.0, abs=1e-8), f"{f} effects must be centred"


def test_intercept_is_the_grand_mean(tmp_path):
    make_grid(tmp_path, token_effect=0.06, cosine_effect=0.02, wide_acc=0.30)
    moe, _ = agg.paired_contrasts(agg.load(tmp_path))
    fit = agg.factorial_fit(moe, "conditional_gain")
    b0 = float(fit.loc[fit.term == "intercept", "estimate"].iloc[0])
    assert b0 == pytest.approx(moe.conditional_gain.mean(), abs=1e-8)


def test_paired_bootstrap_ci_brackets_the_mean_and_sign_test_is_exact():
    st = agg.paired_bootstrap(np.array([0.01, 0.02, 0.03]))
    assert st["n"] == 3
    assert st["mean"] == pytest.approx(0.02)
    assert st["lo"] <= st["mean"] <= st["hi"]
    # three positives out of three: the smallest two-sided p an n=3 sign test can produce
    assert st["p_sign"] == pytest.approx(0.25)
    assert np.isnan(agg.paired_bootstrap(np.array([]))["mean"])


def test_bootstrap_ci_widens_with_noise():
    tight = agg.paired_bootstrap(np.array([0.020, 0.021, 0.019, 0.020, 0.021]))
    loose = agg.paired_bootstrap(np.array([-0.10, 0.30, 0.02, -0.05, 0.15]))
    assert (loose["hi"] - loose["lo"]) > 10 * (tight["hi"] - tight["lo"])


def test_cluster_bootstrap_weights_environments_by_size():
    acc = {0: 1.0, 1: 0.0}
    st = agg.cluster_bootstrap_env(acc, {0: 900, 1: 100}, n_boot=2000)
    assert st["mean"] == pytest.approx(0.9)      # weighted, not the 0.5 unweighted average
    assert st["n_env"] == 2
    assert st["lo"] < st["mean"] < st["hi"]      # resampling 2 clusters is genuinely uncertain
    assert np.isnan(agg.cluster_bootstrap_env({0: 1.0}, {0: 10})["mean"])


# ------------------------------------------------------------------ audits
# ------------------------------------------- per-environment breakdown (the '-1' collapse)
def test_sentinel_only_env_breakdown_is_an_error(tmp_path):
    """The exact regression: OOD runs bucketing every image into site -1."""
    _write(tmp_path, acc_selection=0.3, per_env_val={"-1": 9854}, per_env_n_val={"-1": 9854},
           run_id="rxrx1_moe_collapsed_s0")
    msgs = agg.env_breakdown_audit(agg.load(tmp_path))
    assert any(m.startswith("ERROR") and "-1" in m for m in msgs), msgs


def test_real_multi_env_breakdown_is_silent(tmp_path):
    _write(tmp_path, acc_selection=0.3,
           per_env_val={"33": 0.31, "34": 0.29, "35": 0.30},
           per_env_n_val={"33": 3000, "34": 3400, "35": 3454},
           run_id="rxrx1_moe_ok_s0")
    assert agg.env_breakdown_audit(agg.load(tmp_path)) == []


def test_single_real_env_is_a_warning_not_an_error(tmp_path):
    """Camelyon17's OOD val is one hospital: legitimate, but the bootstrap degenerates."""
    _write(tmp_path, dataset="camelyon17", acc_selection=0.93,
           per_env_val={"1": 0.93}, per_env_n_val={"1": 34904},
           run_id="camelyon17_moe_onehosp_s0")
    msgs = agg.env_breakdown_audit(agg.load(tmp_path))
    assert len(msgs) == 1 and msgs[0].startswith("WARNING"), msgs


def test_env_audit_reaches_the_printed_report(tmp_path):
    """A collapsed breakdown must be visible in the report, not just in the helper."""
    make_grid(tmp_path)
    _write(tmp_path, acc_selection=0.3, per_env_val={"-1": 100}, per_env_n_val={"-1": 100},
           run_id="rxrx1_moe_collapsed_s0")
    assert "single sentinel bucket" in agg.build_report(agg.load(tmp_path))


def test_cluster_bootstrap_is_nan_on_the_collapsed_breakdown():
    """Why the collapse matters: the plan's per-run uncertainty cannot be computed from it."""
    assert np.isnan(agg.cluster_bootstrap_env({-1: 0.3}, {-1: 9854})["mean"])


def test_budget_violation_is_caught(tmp_path):
    _write(tmp_path, variant="moe", seed=0, acc_selection=0.3,
           total_params=30_000_000, run_id="m0")
    _write(tmp_path, variant="dense_wide", seed=0, acc_selection=0.3,
           total_params=31_000_000, run_id="w0")           # +3.3%: far outside the 0.1% tolerance
    lines, ok = agg.budget_audit(agg.load(tmp_path))
    assert not ok and any("VIOLATION" in l for l in lines)


def test_budget_pass_within_tolerance(tmp_path):
    _write(tmp_path, variant="moe", seed=0, acc_selection=0.3,
           total_params=30_000_000, run_id="m0")
    _write(tmp_path, variant="dense_wide", seed=0, acc_selection=0.3,
           total_params=29_996_927, run_id="w0")           # differs by exactly the router
    lines, ok = agg.budget_audit(agg.load(tmp_path))
    assert ok and any("OK" in l for l in lines)


def test_coverage_counts_cells_against_the_predeclared_36(tmp_path):
    make_grid(tmp_path)
    cov = agg.coverage(agg.load(tmp_path))
    assert int(cov.cells_seen.iloc[0]) == 36
    assert int(cov.cells_expected.iloc[0]) == 36
    assert int(cov.moe_runs.iloc[0]) == 108
    assert int(cov.dense_wide.iloc[0]) == 18


# ------------------------------------------------------------------ the stage gate
def test_report_withholds_the_test_split_unless_stage3(tmp_path):
    make_grid(tmp_path)
    _write(tmp_path, variant="moe", seed=0, acc_selection=0.30, acc_heldout=0.99,
           test_evaluated=True, stage=3, run_id="leaky")
    df = agg.load(tmp_path)
    assert "0.99" not in agg.build_report(df, stage3=False), "test accuracy leaked into Stage 1"
    assert "withheld" in agg.build_report(df, stage3=False)
    assert "0.99" in agg.build_report(df, stage3=True)


def test_selection_on_test_split_raises_a_warning(tmp_path):
    _write(tmp_path, variant="moe", seed=0, acc_selection=0.3,
           selection_split="ood_test(no_val_split)", run_id="m0")
    msgs = agg.leakage_audit(agg.load(tmp_path), stage3=False)
    assert any("selected on the test split" in m for m in msgs)


def test_report_runs_end_to_end_on_a_full_grid(tmp_path):
    make_grid(tmp_path, token_effect=0.05, cosine_effect=0.01, noise=0.005)
    rep = agg.build_report(agg.load(tmp_path), stage3=False)
    for section in ["Coverage", "conditional_gain", "Factor effects", "budget audit"]:
        assert section in rep
