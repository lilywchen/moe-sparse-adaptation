#!/usr/bin/env python
"""Aggregate the 12-hour RxRx1 campaign into machine-readable and narrative reports."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _mean_sd(values):
    values = np.asarray([value for value in values if value is not None], dtype=float)
    return {"mean": float(values.mean()) if len(values) else None,
            "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            "n": int(len(values))}


def _combined(result):
    pairs = []
    for acc_key, n_key in (("per_env_val", "per_env_n_val"),
                           ("per_env_heldout", "per_env_n_heldout")):
        for env, accuracy in (result.get(acc_key) or {}).items():
            pairs.append((str(env), float(accuracy), int(result[n_key][str(env)])))
    if not pairs:
        return None
    return {
        "image_weighted": sum(acc * n for _env, acc, n in pairs) / sum(n for _env, _acc, n in pairs),
        "experiment_macro": float(np.mean([acc for _env, acc, _n in pairs])),
        "n_experiments": len(pairs), "n_images": sum(n for _env, _acc, n in pairs),
    }


def _paired_cluster_bootstrap(rows_a, rows_b, draws=10000, seed=20260819):
    by_seed_a = {int(row["seed"]): row for row in rows_a}
    by_seed_b = {int(row["seed"]): row for row in rows_b}
    seeds = sorted(set(by_seed_a) & set(by_seed_b))
    paired = []
    for seed_value in seeds:
        a, b = by_seed_a[seed_value], by_seed_b[seed_value]
        differences = []
        for key in ("per_env_val", "per_env_heldout"):
            aa, bb = a.get(key) or {}, b.get(key) or {}
            for environment in sorted(set(aa) & set(bb)):
                differences.append(float(aa[environment]) - float(bb[environment]))
        paired.append(np.asarray(differences))
    if not paired or not paired[0].size:
        return None
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(draws):
        picked_seeds = rng.integers(0, len(paired), size=len(paired))
        seed_means = []
        for picked in picked_seeds:
            values = paired[picked]
            seed_means.append(values[rng.integers(0, len(values), size=len(values))].mean())
        samples.append(float(np.mean(seed_means)))
    estimate = float(np.mean([values.mean() for values in paired]))
    return {"experiment_macro_difference": estimate,
            "crossed_seed_experiment_bootstrap_ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
            "n_seeds": len(seeds), "n_experiments_per_seed": int(len(paired[0]))}


def aggregate(root):
    manifest = json.loads((root / "campaign_manifest.json").read_text())
    completed, failed = [], []
    for planned in manifest["runs"]:
        result_path = root / f"{planned['run_id']}.json"
        failure_path = root / f"{planned['run_id']}.failure.json"
        if result_path.is_file():
            completed.append({**planned, **json.loads(result_path.read_text())})
        elif failure_path.is_file():
            failed.append({**planned, **json.loads(failure_path.read_text())})

    phase_method = defaultdict(list)
    for row in completed:
        phase_method[(row["phase"], row["label"])].append(row)
    summary = {
        "schema_version": 1, "campaign": manifest["campaign"],
        "complete": len(completed), "failed": failed, "planned": len(manifest["runs"]),
        "confirmatory": {}, "discovery": {}, "replication": {}, "paired_vs_adabn": {},
    }
    for (phase, label), rows in phase_method.items():
        record = {
            "seeds": sorted(int(row["seed"]) for row in rows),
            "ood_val": _mean_sd([row.get("acc_val") for row in rows]),
            "ood_test": _mean_sd([row.get("acc_heldout") for row in rows]),
            "worst_test": _mean_sd([row.get("worst_env_heldout") for row in rows]),
            "id_test": _mean_sd([row.get("acc_within") for row in rows]),
            "wall_hours": float(sum(float(row.get("wall_seconds", 0)) for row in rows) / 3600),
        }
        combined = [_combined(row) for row in rows]
        record["paper_split_18_experiment_image_weighted"] = _mean_sd(
            [value["image_weighted"] for value in combined if value])
        record["paper_split_18_experiment_macro"] = _mean_sd(
            [value["experiment_macro"] for value in combined if value])
        curves = defaultdict(list)
        for row in rows:
            for budget, value in (row.get("context_curve_val") or {}).items():
                curves[budget].append(value["acc_mean"])
        record["validation_context_curve"] = {
            budget: _mean_sd(values) for budget, values in sorted(
                curves.items(), key=lambda pair: int(pair[0]))}
        summary[phase][label] = record

    confirmatory = {label: rows for (phase, label), rows in phase_method.items()
                    if phase == "confirmatory"}
    adabn = confirmatory.get("H2_adabn", [])
    for label, rows in confirmatory.items():
        if label != "H2_adabn":
            summary["paired_vs_adabn"][label] = _paired_cluster_bootstrap(rows, adabn)

    audit_path = root / "hypothesis_audit" / "batch_hypotheses.json"
    summary["hypothesis_audit"] = json.loads(audit_path.read_text()) if audit_path.is_file() else None
    return manifest, summary


def _pct(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def _pm(record):
    if not record or record["mean"] is None:
        return "—"
    suffix = "" if record["sd"] is None else f" ± {100 * record['sd']:.2f}"
    return f"{100 * record['mean']:.2f}{suffix}% (n={record['n']})"


def render(manifest, summary, root):
    lines = ["# RxRx1 batch effects: 12-hour falsification sweep", "",
             f"Status: **{summary['complete']}/{summary['planned']} complete**, "
             f"**{len(summary['failed'])} failed**.  ",
             "Selection remained on the four-experiment OOD validation split; only the four "
             "predeclared core methods received fresh-seed test readout.", "",
             "## Confirmatory results", "",
             "| Method | ID test | OOD val | OOD test | Worst test experiment | "
             "Paper-comparable 18-exp split |", "|---|---:|---:|---:|---:|---:|"]
    order = ["original_grouped", "HarmonyDG", "H2_adabn", "TransportMoE"]
    for label in order:
        row = summary["confirmatory"].get(label, {})
        lines.append(f"| {label} | {_pm(row.get('id_test'))} | {_pm(row.get('ood_val'))} | "
                     f"{_pm(row.get('ood_test'))} | {_pm(row.get('worst_test'))} | "
                     f"{_pm(row.get('paper_split_18_experiment_image_weighted'))} |")
    lines += ["", "The 18-experiment column combines the local 4 validation and 14 test "
              "experiments with image-count weighting. It matches the paper's split membership, "
              "but not its DenseNet-161, 512×512 input, batch size 512, or eight-A100 training "
              "recipe, so it is a reasonableness anchor—not a claimed exact reproduction.", "",
              "Paper anchors: baseline **75.1%**, AdaBN **87.1%** on the batch-separated split; "
              "paper AdaBN cell-line results were HUVEC 92.1, RPE 87.2, HepG2 86.2, U2OS 68.2%.",
              "", "## Paired AdaBN comparisons", ""]
    for label, value in summary["paired_vs_adabn"].items():
        if value is None:
            lines.append(f"- {label}: unavailable")
            continue
        lo, hi = value["crossed_seed_experiment_bootstrap_ci95"]
        estimate = value["experiment_macro_difference"]
        verdict = ("beats AdaBN" if lo > 0 else
                   "non-inferior within 0.5 points" if lo > -0.005 else
                   "does not establish non-inferiority")
        lines.append(f"- {label}: {100*estimate:+.2f} points, crossed seed×experiment 95% CI "
                     f"[{100*lo:+.2f}, {100*hi:+.2f}]; **{verdict}**.")
    lines += ["", "## Target-information budget", "",
              "Validation accuracy using the same confirmatory weights while limiting each "
              "experiment batch to 8, 16, 32, or 64 unlabelled images:", "",
              "| Method | 0 (inductive) | 8 | 16 | 32 | 64 |",
              "|---|---:|---:|---:|---:|---:|"]
    harmony = summary["confirmatory"].get("HarmonyDG", {})
    lines.append(f"| HarmonyDG | {_pm(harmony.get('ood_val'))} | — | — | — | — |")
    for label in ("H2_adabn", "TransportMoE"):
        curve = summary["confirmatory"].get(label, {}).get("validation_context_curve", {})
        lines.append("| " + label + " | — | " + " | ".join(
            _pm(curve.get(str(size))) for size in (8, 16, 32, 64)) + " |")
    lines += ["", "Ambition ladder: **L0** ordinary ERM (no target information); **L1** "
              "HarmonyDG learns from matched source experiments and remains fully inductive; "
              "**L2** AdaBN consumes target moments; **L3** TransportMoE consumes the same moments "
              "but adds learned shared operators; **L4** label-matched target correction is an "
              "analysis ceiling only."]
    lines += ["", "## Discovery and replication", "",
              "| Phase | Method | OOD val | ID test |", "|---|---|---:|---:|"]
    for phase in ("discovery", "replication"):
        for label, row in sorted(summary[phase].items()):
            lines.append(f"| {phase} | {label} | {_pm(row['ood_val'])} | {_pm(row['id_test'])} |")
    audit = summary.get("hypothesis_audit")
    lines += ["", "## Quantitative H0–H4 audit", ""]
    if audit:
        for key, value in audit["hypotheses_supported"].items():
            lines.append(f"- {key}: **{'supported' if value else 'rejected'}**")
        lines.append(f"- Median interaction rank90: {audit['interaction_rank90_median']:.1f}")
        lines.append("- Unlabelled batch-moment descriptor → correction-family coefficient "
                     f"LOO R²: {audit['unlabelled_descriptor_to_operator_coefficient_loo_r2_mean']:.3f}")
        lines.append("- Full geometry report: `hypothesis_audit/batch_hypotheses.md`")
    else:
        lines.append("Geometry audit is still pending.")
    lines += ["", "## Interpretation rules", "",
              "- Success is held-out perturbation prediction, especially the worst experiment; "
              "batch decodability alone is not success.",
              "- `original_random` versus `original_grouped` isolates the batch-ordering change.",
              "- H1→H2→H3→H4 comparisons identify the minimum correction complexity supported.",
              "- The MoE claim requires both a predictive gain and reusable routing: positive "
              "descriptor-to-operator LOO R², multiple effective experts, and a nonzero correction.",
              "- Test results do not choose architecture or hyperparameters; they answer only the "
              "predeclared confirmatory comparison."]
    if summary["failed"]:
        lines += ["", "## Failed runs", ""]
        for row in summary["failed"]:
            lines.append(f"- {row['label']} seed {row['seed']}: `{row['log']}`")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    args = parser.parse_args()
    root = Path(args.result_root).expanduser().resolve()
    manifest, summary = aggregate(root)
    (root / "aggregate.json").write_text(json.dumps(summary, indent=2))
    report = render(manifest, summary, root)
    (root / "REPORT.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
