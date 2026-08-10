#!/usr/bin/env python
"""Paired seed/batch uncertainty for the three-point RxRx1 domain curve."""
import argparse
import json
import math
import statistics
from pathlib import Path

try:
    from scripts.aggregate_rxrx1_domain_scaling_replicate import ARMS, normalized_config
    from scripts.analyze_rxrx1_scaling_uncertainty import (
        _canonical_env_map,
        _weighted,
        hierarchical_batch_bootstrap,
        seed_bootstrap,
    )
except ImportError:  # Direct ``python scripts/...`` execution.
    from aggregate_rxrx1_domain_scaling_replicate import ARMS, normalized_config
    from analyze_rxrx1_scaling_uncertainty import (
        _canonical_env_map,
        _weighted,
        hierarchical_batch_bootstrap,
        seed_bootstrap,
    )


SCALES = (("quarter", 8), ("midpoint", 16), ("full", 33))
CONTRASTS = (
    ("shared-dense", "shared_E3k1_late2", "dense_E4_late2"),
    ("shared-replacement", "shared_E3k1_late2", "replace_E4k2_late2"),
    ("dense-original", "dense_E4_late2", "original"),
)


def _json(path):
    return json.loads(Path(path).read_text())


def _slope(xs, ys):
    xbar = statistics.mean(xs)
    ybar = statistics.mean(ys)
    return sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / sum(
        (x - xbar) ** 2 for x in xs)


def _midpoint_deviation(xs, ys):
    expected = ys[0] + (ys[2] - ys[0]) * (xs[1] - xs[0]) / (xs[2] - xs[0])
    return ys[1] - expected


def load_scale(root, scale, seeds):
    root = Path(root).expanduser().resolve()
    manifest = _json(root / "wave_manifest.json")
    rows = {}
    for spec in manifest["runs"]:
        arm, seed = spec.get("arm"), int(spec.get("seed", -1))
        if arm not in ARMS or seed not in seeds:
            continue
        result = _json(root / f"{spec['run_id']}.json")
        if result.get("run_id") != spec["run_id"]:
            raise ValueError(f"run-id mismatch for {spec['run_id']}")
        if int(result.get("stage", -1)) != 3 or not result.get("test_evaluated"):
            raise ValueError(f"nonterminal result for {spec['run_id']}")
        if result.get("git_dirty"):
            raise ValueError(f"dirty result for {spec['run_id']}")
        rows[(scale, arm, seed)] = result
    expected = {(scale, arm, seed) for arm in ARMS for seed in seeds}
    if set(rows) != expected:
        raise ValueError(f"incomplete {scale} cells: {sorted(expected - set(rows))}")
    return manifest, rows


def summarize_contrast(rows, left, right, seeds, n_boot=20000):
    xs = [math.log2(count) for _scale, count in SCALES]
    gaps_by_seed = {}
    slopes = {}
    deviations = {}
    env_slopes = {}
    env_deviations = {}
    env_counts = {}
    for seed in seeds:
        gaps = []
        per_scale_env = []
        per_scale_counts = []
        for scale, _count in SCALES:
            lrow, rrow = rows[(scale, left, seed)], rows[(scale, right, seed)]
            gaps.append(100.0 * (lrow["acc_heldout"] - rrow["acc_heldout"]))
            lacc = _canonical_env_map(lrow, "per_env_heldout")
            racc = _canonical_env_map(rrow, "per_env_heldout")
            lcounts = _canonical_env_map(lrow, "per_env_n_heldout")
            rcounts = _canonical_env_map(rrow, "per_env_n_heldout")
            if set(lacc) != set(racc) or set(lacc) != set(lcounts) or lcounts != rcounts:
                raise ValueError(f"held-out batch drift for {scale}/{seed}/{left}-{right}")
            per_scale_env.append({env: 100.0 * (lacc[env] - racc[env]) for env in lacc})
            per_scale_counts.append(lcounts)
        if any(counts != per_scale_counts[0] for counts in per_scale_counts[1:]):
            raise ValueError(f"held-out denominators change across scale for seed {seed}")
        gaps_by_seed[seed] = dict(zip((scale for scale, _ in SCALES), gaps))
        slopes[seed] = _slope(xs, gaps)
        deviations[seed] = _midpoint_deviation(xs, gaps)
        envs = sorted(per_scale_env[0], key=str)
        env_slopes[seed] = {
            env: _slope(xs, [value[env] for value in per_scale_env]) for env in envs}
        env_deviations[seed] = {
            env: _midpoint_deviation(xs, [value[env] for value in per_scale_env]) for env in envs}
        env_counts[seed] = per_scale_counts[0]
        recovered = _weighted([env_slopes[seed][env] for env in envs],
                              [env_counts[seed][env] for env in envs])
        if abs(recovered - slopes[seed]) > 0.002:
            raise ValueError(f"batch decomposition does not recover slope for seed {seed}")

    mean_gaps = {
        scale: statistics.mean(gaps_by_seed[seed][scale] for seed in seeds)
        for scale, _count in SCALES
    }
    batch_means = {
        env: statistics.mean(env_slopes[seed][env] for seed in seeds)
        for env in sorted(env_slopes[seeds[0]], key=str)
    }
    return {
        "per_seed_gap_points": gaps_by_seed,
        "mean_gap_points": mean_gaps,
        "per_seed_slope_points_per_octave": slopes,
        "per_seed_midpoint_deviation_points": deviations,
        "slope_seed_bootstrap": seed_bootstrap(slopes, n_boot=n_boot),
        "slope_hierarchical_seed_batch_bootstrap": hierarchical_batch_bootstrap(
            env_slopes, env_counts, n_boot=n_boot),
        "midpoint_deviation_seed_bootstrap": seed_bootstrap(deviations, n_boot=n_boot),
        "midpoint_deviation_hierarchical_seed_batch_bootstrap": hierarchical_batch_bootstrap(
            env_deviations, env_counts, n_boot=n_boot),
        "positive_batch_slope_fraction": sum(value > 0 for value in batch_means.values()) / len(batch_means),
        "per_batch_mean_slope_points_per_octave": batch_means,
    }


def analyze(quarter_root, midpoint_root, full_root, n_boot=20000):
    midpoint_manifest = _json(Path(midpoint_root) / "wave_manifest.json")
    seeds = tuple(map(int, midpoint_manifest["seeds"]))
    manifests = {}
    rows = {}
    for scale, root in (("quarter", quarter_root), ("midpoint", midpoint_root), ("full", full_root)):
        manifests[scale], loaded = load_scale(root, scale, seeds)
        rows.update(loaded)
    for arm in ARMS:
        for seed in seeds:
            configs = [normalized_config(rows[(scale, arm, seed)]["config"]) for scale, _ in SCALES]
            if any(config != configs[0] for config in configs[1:]):
                raise ValueError(f"resolved-config drift for {arm}/seed{seed}")
    return {
        "scales": {scale: count for scale, count in SCALES},
        "seeds": list(seeds),
        "n_boot": int(n_boot),
        "contrasts": {
            label: summarize_contrast(rows, left, right, seeds, n_boot=n_boot)
            for label, left, right in CONTRASTS
        },
    }


def render_markdown(report):
    lines = [
        "# RxRx1 three-point domain-scaling uncertainty",
        "",
        "Slopes are architecture-gap percentage points per doubling of training experiments.",
        "The hierarchical CI resamples paired seeds and held-out test batches; with only two seeds it remains diagnostic.",
        "",
        "| Contrast | Gap @8 | Gap @16 | Gap @33 | Seed slopes | Mean slope | Seed 95% CI | Seed+batch 95% CI | Positive batches | Midpoint deviation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, summary in report["contrasts"].items():
        seed = summary["slope_seed_bootstrap"]
        hierarchy = summary["slope_hierarchical_seed_batch_bootstrap"]
        deviation = summary["midpoint_deviation_seed_bootstrap"]
        values = "/".join(f"{value:+.3f}" for _, value in sorted(
            summary["per_seed_slope_points_per_octave"].items()))
        gaps = summary["mean_gap_points"]
        lines.append(
            f"| {label} | {gaps['quarter']:+.3f} | {gaps['midpoint']:+.3f} | {gaps['full']:+.3f} | "
            f"{values} | {seed['mean']:+.3f} | [{seed['lo']:+.3f}, {seed['hi']:+.3f}] | "
            f"[{hierarchy['lo']:+.3f}, {hierarchy['hi']:+.3f}] | "
            f"{100.0 * summary['positive_batch_slope_fraction']:.1f}% | {deviation['mean']:+.3f} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("midpoint_root")
    parser.add_argument("--quarter-root", required=True)
    parser.add_argument("--full-root", required=True)
    parser.add_argument("--n-boot", type=int, default=20000)
    parser.add_argument("--output-json")
    parser.add_argument("--output-markdown")
    args = parser.parse_args()
    report = analyze(args.quarter_root, args.midpoint_root, args.full_root, args.n_boot)
    markdown = render_markdown(report)
    print(markdown, end="")
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(report, indent=2, sort_keys=True))
    if args.output_markdown:
        Path(args.output_markdown).write_text(markdown)


if __name__ == "__main__":
    main()
