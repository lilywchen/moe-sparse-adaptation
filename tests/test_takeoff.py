"""Guards on the epoch-budget diagnostic.

The curves here all carry a WARMUP TRANSIENT, because the first version of this diagnostic passed
a clean synthetic suite and was still wrong on every real run. Training uses a 5-epoch LR warmup,
so the loss falls off its random-init value very fast around epoch 3; that spike is the steepest
part of every curve. The original check looked for the steepest point, found the transient, and
concluded that everything after it was flat - reporting the known-plateaued RxRx1 phase-A runs as
CONVERGED. A synthetic curve without a warmup spike cannot catch that, which is why every fixture
below has one. Torch-free.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.takeoff_check import analyse, read_trainlog, smooth  # noqa: E402

WARMUP = 5


def with_warmup(tail, init=7.9, warmup=WARMUP):
    """Prepend the fast drop from random init that a 5-epoch LR warmup produces.

    tail is [(epoch, loss)] for the post-warmup curve; the transient is interpolated from `init`
    down to the tail's first value, which is the shape every real trainlog starts with.
    """
    start = tail[0][1]
    head = [(e, init - (init - start) * (e / warmup)) for e in range(warmup)]
    return head + [(e + warmup, l) for e, l in tail]


def converged_tail(n, floor=0.5, start=7.0, rate=0.08):
    """A converged run: descent that flattens well before the budget runs out."""
    return [(e, floor + (start - floor) * math.exp(-rate * e)) for e in range(n)]


def plateau_tail(n, start=7.04, accel=0.0016):
    """A pre-takeoff run: quadratic, so each epoch removes more loss than the last."""
    return [(e, start - accel * e * e) for e in range(n)]


def test_converged_curve_is_called_converged():
    r = analyse(with_warmup(converged_tail(85)))
    assert r["verdict"] == "CONVERGED", r


def test_still_accelerating_curve_is_called_pre_takeoff():
    r = analyse(with_warmup(plateau_tail(25)))
    assert r["verdict"] == "PRE-TAKEOFF", r
    assert r["late_vs_mid"] >= 0.9


def test_the_warmup_transient_does_not_mask_a_plateau():
    """The regression this whole module exists for.

    A plateau preceded by a large warmup spike must still read PRE-TAKEOFF. Judged against the
    spike it looks flat, which is what the previous implementation did and why phase A slipped
    through.
    """
    curve = with_warmup(plateau_tail(25), init=7.9)
    # The transient really is the steepest part of the curve, i.e. the trap is present.
    drops = [curve[i - 1][1] - curve[i][1] for i in range(1, len(curve))]
    assert max(drops[:WARMUP]) > max(drops[WARMUP:]), "fixture must contain the warmup trap"
    assert analyse(curve)["verdict"] == "PRE-TAKEOFF"


def test_the_real_rxrx1_phase_a_shape_is_flagged():
    """30 epochs, init ~7.7 -> ~5.9, never bending over: 16% of the way down from uniform."""
    curve = with_warmup(plateau_tail(25), init=7.66)
    assert 5.5 < curve[-1][1] < 6.5, "sanity: should land in the observed 5.78-6.59 band"
    r = analyse(curve, classes=1139)
    assert r["verdict"] == "PRE-TAKEOFF"
    assert r["frac_of_uniform_removed"] < 0.2


def test_a_run_that_is_slowing_but_still_working_is_marginal():
    r = analyse(with_warmup(converged_tail(55, rate=0.022)))
    assert r["verdict"] == "MARGINAL", r


def test_short_curves_refuse_to_answer_rather_than_guess():
    r = analyse(converged_tail(6))
    assert r["verdict"] == "TOO-SHORT"
    assert "epochs" in r["reason"]


def test_a_flat_or_rising_curve_is_not_called_a_budget_problem():
    """Divergence is a different pathology; saying 'train longer' would be actively wrong."""
    flat = [(e, 6.9 + 0.001 * e) for e in range(40)]
    r = analyse(flat)
    assert r["verdict"] == "MARGINAL"
    assert "LR" in r["reason"] or "divergence" in r["reason"]


def test_uniform_baseline_is_reported_against_the_class_count():
    r = analyse(with_warmup(converged_tail(85)), classes=1139)
    assert abs(r["uniform_loss"] - math.log(1139)) < 1e-9
    assert 0.0 < r["frac_of_uniform_removed"] <= 1.0


def test_noise_does_not_flip_the_verdict():
    """A finite difference on raw epoch losses flips sign constantly; smoothing is load-bearing."""
    import random
    random.seed(0)
    noisy_conv = [(e, l + random.uniform(-0.05, 0.05))
                  for e, l in with_warmup(converged_tail(85))]
    assert analyse(noisy_conv)["verdict"] == "CONVERGED"
    noisy_plateau = [(e, l + random.uniform(-0.05, 0.05))
                     for e, l in with_warmup(plateau_tail(25))]
    assert analyse(noisy_plateau)["verdict"] == "PRE-TAKEOFF"


def test_smooth_preserves_length():
    vals = [float(v) for v in range(20)]
    sm = smooth(vals)
    assert len(sm) == len(vals)
    assert sm[0] < sm[-1]


def test_reader_survives_a_truncated_final_line(tmp_path):
    """Trainlogs are read while the job is still writing, so a half-line is the normal case."""
    p = tmp_path / "x.trainlog.jsonl"
    p.write_text('{"epoch": 1, "loss": 7.0}\n{"epoch": 2, "loss": 6.5}\n{"epoch": 3, "los')
    assert read_trainlog(p) == [(1, 7.0), (2, 6.5)]


def test_reader_keeps_the_last_record_for_a_repeated_epoch(tmp_path):
    """A resumed run re-emits epochs; the later record is the live one."""
    p = tmp_path / "y.trainlog.jsonl"
    p.write_text(json.dumps({"epoch": 1, "loss": 7.0}) + "\n"
                 + json.dumps({"epoch": 1, "loss": 6.0}) + "\n")
    assert read_trainlog(p) == [(1, 6.0)]
