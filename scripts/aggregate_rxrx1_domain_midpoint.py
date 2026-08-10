#!/usr/bin/env python
"""One-command audit/table for the three-point RxRx1 domain-count curve."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

try:
    from scripts.aggregate_rxrx1_domain_scaling_replicate import (
        ARMS, CONTRASTS, _pct, _points, load_wave, normalized_config,
        validate_artifacts,
    )
except ImportError:  # Direct ``python scripts/...`` execution.
    from aggregate_rxrx1_domain_scaling_replicate import (
        ARMS, CONTRASTS, _pct, _points, load_wave, normalized_config,
        validate_artifacts,
    )


SCALE_LABELS = (("quarter", 8), ("midpoint", 16), ("full", 33))


def _mean_sem(values):
    values = [float(value) for value in values]
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, None
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance / len(values))


def _slope(xs, ys):
    xbar = sum(xs) / len(xs)
    ybar = sum(ys) / len(ys)
    return sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / sum(
        (x - xbar) ** 2 for x in xs)


def _load_scale(root, scale, expected_environments=None):
    root, manifest, rows = load_wave(root)
    problems = validate_artifacts(manifest, rows, expected_environments)
    mapping = {(row["arm"], int(row["seed"])): row for row in rows if row["arm"] in ARMS}
    return root, manifest, rows, mapping, problems


def render_report(midpoint_root, quarter_root=None, full_root=None):
    mroot = Path(midpoint_root).expanduser().resolve()
    mmanifest = json.loads((mroot / "wave_manifest.json").read_text())
    quarter_root = quarter_root or mmanifest["quarter_anchor_root"]
    full_root = full_root or mmanifest["full_anchor_root"]
    q = _load_scale(quarter_root, "quarter", mmanifest["quarter_environment_subset"])
    m = _load_scale(mroot, "midpoint", mmanifest["midpoint_environment_subset"])
    f = _load_scale(full_root, "full", None)
    scales = {"quarter": q, "midpoint": m, "full": f}
    problems = [f"{scale}: {problem}" for scale, value in scales.items()
                for problem in value[4]]

    seeds = tuple(map(int, mmanifest.get("seeds", [])))
    for arm in ARMS:
        for seed in seeds:
            cells = [scales[scale][3].get((arm, seed)) for scale, _count in SCALE_LABELS]
            if any(cell is None for cell in cells):
                problems.append(f"missing curve cell for {arm}/seed{seed}")
                continue
            completed = [cell for cell in cells if cell.get("result")]
            if len(completed) < 2:
                continue
            reference = normalized_config(completed[0]["result"]["config"])
            if any(normalized_config(cell["result"]["config"]) != reference
                   for cell in completed[1:]):
                problems.append(f"resolved-config drift for {arm}/seed{seed}")

    complete = sum(row["state"] == "complete" for row in m[2])
    lines = [f"{mmanifest.get('campaign', mroot.name)} — {complete}/{len(m[2])} new rows complete"]
    lines.append("Protocol: " + ("; ".join(problems) if problems else
                                  "all available artifacts and three-point configs pass"))
    lines += [
        "| Scale | Seed | Arm | State | Ep | Train | raw ID* | OOD val | OOD test | Worst test |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scale, _count in SCALE_LABELS:
        for seed in seeds:
            for arm in ARMS:
                row = scales[scale][3].get((arm, seed))
                if row is None:
                    continue
                lines.append(
                    f"| {scale} | {seed} | {arm} | {row['state']} | {row['epoch']} | "
                    f"{_pct(row['acc_train'])} | {_pct(row['acc_within'])} | "
                    f"{_pct(row['acc_val'])} | {_pct(row['acc_heldout'])} | "
                    f"{_pct(row['worst_env_heldout'])} |")

    lines += ["", "Architecture-gap slopes across log2(training experiments), points/octave:",
              "| Contrast | Metric | Seed 1 | Seed 2 | Mean | SEM |",
              "|---|---|---:|---:|---:|---:|"]
    slopes = defaultdict(dict)
    for left, right in CONTRASTS:
        for metric in ("acc_heldout", "worst_env_heldout", "acc_val"):
            for seed in seeds:
                values = []
                for scale, count in SCALE_LABELS:
                    left_row = scales[scale][3].get((left, seed))
                    right_row = scales[scale][3].get((right, seed))
                    if (left_row is None or right_row is None or
                            left_row.get(metric) is None or right_row.get(metric) is None):
                        break
                    values.append(100.0 * (left_row[metric] - right_row[metric]))
                if len(values) == len(SCALE_LABELS):
                    slopes[(left, right, metric)][seed] = _slope(
                        [math.log2(count) for _scale, count in SCALE_LABELS], values)
            per_seed = slopes[(left, right, metric)]
            mean, sem = _mean_sem(per_seed.values())
            label = {"acc_heldout": "OOD test", "worst_env_heldout": "Worst test",
                     "acc_val": "OOD val"}[metric]
            rendered = [_points(per_seed.get(seed)) for seed in seeds]
            lines.append(f"| {left} − {right} | {label} | " + " | ".join(rendered) +
                         f" | {_points(mean)} | {_points(sem)} |")

    lines.append("")
    lines.append("* raw ID is not a scale-comparable endpoint until fixed-environment re-evaluation.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("midpoint_root")
    parser.add_argument("--quarter-root")
    parser.add_argument("--full-root")
    args = parser.parse_args()
    print(render_report(args.midpoint_root, args.quarter_root, args.full_root))


if __name__ == "__main__":
    main()
