#!/usr/bin/env python
"""Paired seed/batch uncertainty for the completed RxRx1 domain-scaling curve."""
import argparse
import json
import random
import statistics
from pathlib import Path


ARMS = ("original", "dense_E4_late2", "replace_E4k2_late2", "shared_E3k1_late2")
CONTRASTS = (
    ("shared-dense", "shared_E3k1_late2", "dense_E4_late2"),
    ("shared-replacement", "shared_E3k1_late2", "replace_E4k2_late2"),
    ("dense-original", "dense_E4_late2", "original"),
)


def _json(path):
    return json.loads(Path(path).read_text())


def _percentile(values, q):
    values = sorted(float(value) for value in values)
    if not values:
        return float("nan")
    position = (len(values) - 1) * float(q)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _weighted(values, weights):
    denominator = sum(float(weight) for weight in weights)
    if denominator <= 0:
        raise ValueError("batch weights must be positive")
    return sum(float(value) * float(weight) for value, weight in zip(values, weights)) / denominator


def seed_bootstrap(per_seed, n_boot=10000, rng_seed=20260810):
    seeds = sorted(per_seed)
    if len(seeds) < 2:
        return {"mean": statistics.mean(per_seed.values()), "lo": None, "hi": None}
    rng = random.Random(rng_seed)
    draws = []
    for _ in range(int(n_boot)):
        draws.append(statistics.mean(per_seed[rng.choice(seeds)] for _ in seeds))
    return {
        "mean": statistics.mean(per_seed.values()),
        "lo": _percentile(draws, 0.025),
        "hi": _percentile(draws, 0.975),
    }


def hierarchical_batch_bootstrap(per_seed_env, per_seed_counts, n_boot=10000,
                                 rng_seed=20260810):
    """Resample paired seeds and test batches while preserving every within-cell pairing."""
    seeds = sorted(per_seed_env)
    if len(seeds) < 2:
        raise ValueError("hierarchical bootstrap requires at least two seeds")
    envs = sorted(per_seed_env[seeds[0]], key=str)
    if len(envs) < 2:
        raise ValueError("hierarchical bootstrap requires at least two test batches")
    for seed in seeds:
        if sorted(per_seed_env[seed], key=str) != envs:
            raise ValueError("test-batch keys differ across seeds")
        if sorted(per_seed_counts[seed], key=str) != envs:
            raise ValueError("test-batch count keys differ across seeds")
    rng = random.Random(rng_seed)
    draws = []
    for _ in range(int(n_boot)):
        sampled_seeds = [rng.choice(seeds) for _ in seeds]
        sampled_envs = [rng.choice(envs) for _ in envs]
        seed_means = []
        for seed in sampled_seeds:
            values = [per_seed_env[seed][environment] for environment in sampled_envs]
            weights = [per_seed_counts[seed][environment] for environment in sampled_envs]
            seed_means.append(_weighted(values, weights))
        draws.append(statistics.mean(seed_means))
    observed = statistics.mean(
        _weighted([per_seed_env[seed][env] for env in envs],
                  [per_seed_counts[seed][env] for env in envs])
        for seed in seeds)
    return {
        "mean": observed,
        "lo": _percentile(draws, 0.025),
        "hi": _percentile(draws, 0.975),
        "n_seeds": len(seeds),
        "n_batches": len(envs),
    }


def load_manifest_results(root, forced_scale=None):
    root = Path(root).expanduser().resolve()
    manifest = _json(root / "wave_manifest.json")
    rows = {}
    for spec in manifest["runs"]:
        if spec.get("arm") not in ARMS:
            continue
        scale = forced_scale or spec.get("scale")
        if scale not in ("quarter", "full"):
            continue
        result = _json(root / f"{spec['run_id']}.json")
        if result.get("run_id") != spec["run_id"]:
            raise ValueError(f"run-id mismatch for {spec['run_id']}")
        if int(result.get("stage", -1)) != 3 or not result.get("test_evaluated"):
            raise ValueError(f"nonterminal result for {spec['run_id']}")
        rows[(scale, spec["arm"], int(spec["seed"]))] = result
    return manifest, rows


def _canonical_env_map(result, key):
    value = result.get(key)
    if not value:
        raise ValueError(f"missing {key} in {result.get('run_id')}")
    return {str(environment): float(score) for environment, score in value.items()}


def summarize_contrast(rows, left, right, n_boot=10000):
    seeds = sorted({key[2] for key in rows if key[0] == "quarter" and key[1] == left}
                   & {key[2] for key in rows if key[0] == "full" and key[1] == left})
    per_seed_overall = {}
    per_seed_quarter_gap = {}
    per_seed_full_gap = {}
    per_seed_env = {}
    per_seed_counts = {}
    for seed in seeds:
        qleft, qright = rows[("quarter", left, seed)], rows[("quarter", right, seed)]
        fleft, fright = rows[("full", left, seed)], rows[("full", right, seed)]
        quarter_gap = 100.0 * (qleft["acc_heldout"] - qright["acc_heldout"])
        full_gap = 100.0 * (fleft["acc_heldout"] - fright["acc_heldout"])
        per_seed_quarter_gap[seed] = quarter_gap
        per_seed_full_gap[seed] = full_gap
        per_seed_overall[seed] = full_gap - quarter_gap

        models = (qleft, qright, fleft, fright)
        acc = [_canonical_env_map(model, "per_env_heldout") for model in models]
        counts = [_canonical_env_map(model, "per_env_n_heldout") for model in models]
        envs = set(acc[0])
        if any(set(value) != envs for value in (*acc[1:], *counts)):
            raise ValueError(f"held-out batch keys drift at seed {seed}")
        for environment in envs:
            observed_counts = {value[environment] for value in counts}
            if len(observed_counts) != 1:
                raise ValueError(f"held-out batch denominator drift at seed {seed}/{environment}")
        per_seed_env[seed] = {
            env: 100.0 * ((acc[2][env] - acc[3][env]) - (acc[0][env] - acc[1][env]))
            for env in envs
        }
        per_seed_counts[seed] = counts[0]
        weighted = _weighted([per_seed_env[seed][env] for env in sorted(envs)],
                             [per_seed_counts[seed][env] for env in sorted(envs)])
        if abs(weighted - per_seed_overall[seed]) > 0.002:
            raise ValueError(f"batch decomposition does not recover overall interaction at seed {seed}")

    batch_means = {
        env: statistics.mean(per_seed_env[seed][env] for seed in seeds)
        for env in sorted(per_seed_env[seeds[0]], key=str)
    }
    batch_values = sorted(batch_means.values())
    return {
        "per_seed_interaction_points": per_seed_overall,
        "per_seed_quarter_gap_points": per_seed_quarter_gap,
        "per_seed_full_gap_points": per_seed_full_gap,
        "quarter_gap_mean_points": statistics.mean(per_seed_quarter_gap.values()),
        "full_gap_mean_points": statistics.mean(per_seed_full_gap.values()),
        "seed_bootstrap": seed_bootstrap(per_seed_overall, n_boot=n_boot),
        "hierarchical_seed_batch_bootstrap": hierarchical_batch_bootstrap(
            per_seed_env, per_seed_counts, n_boot=n_boot),
        "positive_batch_fraction": sum(value > 0 for value in batch_values) / len(batch_values),
        "batch_interaction_min_median_max_points": [
            min(batch_values), statistics.median(batch_values), max(batch_values)],
        "per_batch_mean_interaction_points": batch_means,
    }


def analyze(replicate_root, full_root, seed5_root=None, n_boot=10000):
    _manifest, quarter = load_manifest_results(replicate_root, forced_scale="quarter")
    _manifest, full = load_manifest_results(full_root, forced_scale="full")
    rows = {**quarter, **full}
    if seed5_root:
        _manifest, seed5 = load_manifest_results(seed5_root)
        rows.update(seed5)
    expected = {(scale, arm, seed) for scale in ("quarter", "full")
                for arm in ARMS for seed in sorted({key[2] for key in rows})}
    missing = expected - set(rows)
    if missing:
        raise ValueError(f"incomplete scaling cells: {sorted(missing)}")
    return {
        "seeds": sorted({key[2] for key in rows}),
        "n_boot": int(n_boot),
        "contrasts": {
            label: summarize_contrast(rows, left, right, n_boot=n_boot)
            for label, left, right in CONTRASTS
        },
    }


def render_markdown(report):
    lines = [
        "# RxRx1 domain-scaling paired uncertainty",
        "",
        "Interactions are `(full architecture gap) - (quarter architecture gap)` in percentage points.",
        "The hierarchical CI resamples paired seeds and held-out test batches.",
        "",
        "| Contrast | Seeds | Quarter gap | Full gap | Interaction | Seed 95% CI | Seed+batch 95% CI | Positive batches |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, summary in report["contrasts"].items():
        seed = summary["seed_bootstrap"]
        hierarchy = summary["hierarchical_seed_batch_bootstrap"]
        values = "/".join(f"{value:+.3f}" for _, value in
                          sorted(summary["per_seed_interaction_points"].items()))
        lines.append(
            f"| {label} | {values} | {summary['quarter_gap_mean_points']:+.3f} | "
            f"{summary['full_gap_mean_points']:+.3f} | {seed['mean']:+.3f} | "
            f"[{seed['lo']:+.3f}, {seed['hi']:+.3f}] | "
            f"[{hierarchy['lo']:+.3f}, {hierarchy['hi']:+.3f}] | "
            f"{100.0 * summary['positive_batch_fraction']:.1f}% |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("replicate_root")
    parser.add_argument("--full-root", required=True)
    parser.add_argument("--seed5-root")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--output-json")
    parser.add_argument("--output-markdown")
    args = parser.parse_args()
    report = analyze(args.replicate_root, args.full_root, args.seed5_root, args.n_boot)
    markdown = render_markdown(report)
    print(markdown, end="")
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(report, indent=2, sort_keys=True))
    if args.output_markdown:
        Path(args.output_markdown).write_text(markdown)


if __name__ == "__main__":
    main()
