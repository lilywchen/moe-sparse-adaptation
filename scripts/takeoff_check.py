#!/usr/bin/env python
"""Is this run's epoch budget large enough for its ranking to mean anything?

RxRx1 phase A cost six runs and produced a hyperparameter ranking that had to be discarded. The
reason was not a bug. Every cell was stopped at 30 epochs, and at 30 epochs a DINOv2 ViT-S/14 on
1,139-way RxRx1 is still on the pre-takeoff plateau: train loss 5.78-6.59 against a
uniform-prediction loss of ln(1139) = 7.04, ID accuracy under 4.2%. Ranking there measures *which
cell leaves the plateau first*, which is not the same quantity as *which cell ends up best*, and
nothing in the pipeline said so. The 90-epoch probe made the gap concrete: the same recipe reaches
loss 3.56 by epoch 52 and is still descending at 0.1/epoch.

The tell is in the curve's second derivative: while a run is still on the plateau its descent is
*accelerating*, so the loss removed in the last stretch of training is at least as large as the
loss removed in the stretch before it. A converged run is the opposite - its late stretch
contributes far less than its middle one.

    late_vs_mid = (per-epoch descent over the final third) / (per-epoch descent over the middle third)

The first third is deliberately thrown away, and that detail is load-bearing. An earlier version of
this script looked for the *steepest* point of the whole curve instead, and it was wrong on every
real run: with a 5-epoch LR warmup the steepest descent is always the warmup transient around epoch
3, where the loss falls off its random-init value. Measured against that spike everything later
looks flat, so the script confidently reported the known-plateaued phase-A runs as CONVERGED - the
exact false reassurance it exists to prevent. Dropping the first third removes the transient and
compares only the two stretches that carry information about where training is heading.

Torch-free and stdlib-only, so it runs on the cluster, in the sandbox, and in CI.

    python scripts/takeoff_check.py hpo/rxrx1/phase_a/*.trainlog.jsonl --classes 1139
    python scripts/takeoff_check.py --root hpo --classes 1139
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import sys
from pathlib import Path

# A run is PRE-TAKEOFF when its final third descends at least this fraction as fast as its middle
# third: the descent has not begun to slow, so training stopped while still speeding up.
PRETAKEOFF_RATIO = 0.90
# ...and MARGINAL down to here: visibly slowing, but the tail is still doing real work.
MARGINAL_RATIO = 0.35
# Rolling-mean window. Epoch losses are noisy enough that a raw finite difference flips sign
# constantly; smoothing first is what makes the trend readable at all.
SMOOTH_WINDOW = 5
# Need enough epochs to cut three meaningful thirds.
MIN_EPOCHS = 9

VERDICTS = ("PRE-TAKEOFF", "MARGINAL", "CONVERGED", "TOO-SHORT")


def read_trainlog(path):
    """-> [(epoch, loss)] sorted by epoch, keeping the last record for a repeated epoch."""
    by_epoch = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # a truncated final line is normal while a job is still writing
        if "epoch" in rec and "loss" in rec and rec["loss"] is not None:
            by_epoch[int(rec["epoch"])] = float(rec["loss"])
    return sorted(by_epoch.items())


def smooth(values, window=SMOOTH_WINDOW):
    """Centred rolling mean, shrinking at the edges so length is preserved."""
    if window <= 1 or len(values) < 2:
        return list(values)
    half = window // 2
    out = []
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def analyse(curve, classes=None):
    """curve: [(epoch, loss)] -> dict with the verdict and the numbers behind it."""
    n = len(curve)
    if n < MIN_EPOCHS:
        return {"verdict": "TOO-SHORT", "n_epochs": n,
                "reason": f"only {n} epochs; need >= {MIN_EPOCHS} to read a trend"}

    epochs = [e for e, _ in curve]
    losses = [l for _, l in curve]
    sm = smooth(losses)

    # Thirds by index. The first is discarded because it holds the LR warmup transient; see the
    # module docstring for what happens when it is not.
    i1, i2 = n // 3, 2 * n // 3
    mid_span, late_span = i2 - i1, (n - 1) - i2
    if mid_span < 1 or late_span < 1:
        return {"verdict": "TOO-SHORT", "n_epochs": n,
                "reason": f"only {n} epochs; thirds are too small to compare"}

    mid_drop = sm[i1] - sm[i2]
    late_drop = sm[i2] - sm[-1]
    mid_rate = mid_drop / mid_span
    late_rate = late_drop / late_span
    ratio = (late_rate / mid_rate) if mid_rate > 0 else float("inf")

    if mid_rate <= 0 and late_rate <= 0:
        verdict = "MARGINAL"      # loss not falling on net; a different pathology entirely
        reason = ("smoothed loss is flat or rising after the first third; diagnose LR / "
                  "divergence rather than the epoch budget")
    elif ratio >= PRETAKEOFF_RATIO:
        verdict = "PRE-TAKEOFF"
        reason = (f"final third still descends at {ratio:.0%} of the middle third's rate "
                  f"({late_rate:.4f} vs {mid_rate:.4f} per epoch) - not yet slowing")
    elif ratio >= MARGINAL_RATIO:
        verdict = "MARGINAL"
        reason = (f"descent is slowing but the final third is still {ratio:.0%} of the middle "
                  f"third ({late_rate:.4f} vs {mid_rate:.4f} per epoch)")
    else:
        verdict = "CONVERGED"
        reason = (f"final third down to {ratio:.0%} of the middle third "
                  f"({late_rate:.4f} vs {mid_rate:.4f} per epoch)")

    res = {"verdict": verdict, "reason": reason, "n_epochs": n,
           "first_epoch": epochs[0], "last_epoch": epochs[-1],
           "first_loss": losses[0], "final_loss": losses[-1],
           "mid_rate": mid_rate, "late_rate": late_rate, "late_vs_mid": ratio,
           "split_epochs": (epochs[i1], epochs[i2])}

    if classes:
        import math
        uniform = math.log(classes)
        res["uniform_loss"] = uniform
        # How far along the road from "predicts nothing" to "predicts perfectly" is this run?
        res["frac_of_uniform_removed"] = (uniform - losses[-1]) / uniform
    return res


def format_row(name, r):
    if r["verdict"] == "TOO-SHORT":
        return f"  {r['verdict']:<12} {name}\n                 {r['reason']}"
    tail = ""
    if "frac_of_uniform_removed" in r:
        tail = (f"  |  {r['frac_of_uniform_removed']:.0%} of the way down from "
                f"uniform ({r['uniform_loss']:.2f})")
    return (f"  {r['verdict']:<12} {name}\n"
            f"                 ep {r['first_epoch']}-{r['last_epoch']}  "
            f"loss {r['first_loss']:.3f} -> {r['final_loss']:.3f}{tail}\n"
            f"                 {r['reason']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="*", help="*.trainlog.jsonl files")
    ap.add_argument("--root", default=None,
                    help="directory to search recursively for *.trainlog.jsonl")
    ap.add_argument("--classes", type=int, default=None,
                    help="number of classes, to report loss against the uniform baseline "
                         "(rxrx1: 1139, camelyon17: 2)")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--fail-on-pretakeoff", action="store_true",
                    help="exit 1 if any run is still pre-takeoff (for use in a gate)")
    args = ap.parse_args(argv)

    paths = list(args.paths)
    if args.root:
        paths += sorted(globmod.glob(os.path.join(args.root, "**", "*.trainlog.jsonl"),
                                     recursive=True))
    if not paths:
        ap.error("no trainlogs given; pass paths or --root")

    results = {}
    for p in paths:
        curve = read_trainlog(p)
        if not curve:
            continue
        results[p] = analyse(curve, args.classes)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        order = {v: i for i, v in enumerate(VERDICTS)}
        print(f"takeoff check: {len(results)} run(s)\n")
        for p, r in sorted(results.items(), key=lambda kv: (order[kv[1]["verdict"]], kv[0])):
            print(format_row(Path(p).name.replace(".trainlog.jsonl", ""), r))
        counts = {}
        for r in results.values():
            counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        print("\n  " + "  ".join(f"{v}={counts.get(v, 0)}" for v in VERDICTS))
        if counts.get("PRE-TAKEOFF"):
            print("\n  PRE-TAKEOFF runs were stopped while still speeding up. Any ranking over\n"
                  "  them orders plateau-exit speed, not final quality. Raise train.epochs and\n"
                  "  re-run before reading anything off the comparison.")

    if args.fail_on_pretakeoff and any(r["verdict"] == "PRE-TAKEOFF" for r in results.values()):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
