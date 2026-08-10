#!/usr/bin/env python
"""One-command status for the locked RxRx1 domain-scaling replication.

The new wave trains only the eight-experiment endpoint at seeds 1/2 and joins it to the
already-completed, protocol-matched full-data anchors from ``shared_confirm30_20260809``.
"""
import argparse
import copy
import json
import math
from collections import defaultdict
from pathlib import Path


METRICS = ("acc_train", "acc_within", "acc_val", "acc_heldout", "worst_env_heldout")
ARMS = ("original", "dense_E4_late2", "replace_E4k2_late2", "shared_E3k1_late2")
CONTRASTS = (
    ("shared_E3k1_late2", "dense_E4_late2"),
    ("shared_E3k1_late2", "replace_E4k2_late2"),
    ("dense_E4_late2", "original"),
)


def _json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def _last_jsonl(path):
    try:
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        return rows[-1] if rows else None
    except Exception:
        return None


def _metric(result, milestone, key):
    value = (result or {}).get(key)
    if value is None:
        value = (milestone or {}).get(key)
    if value is None and key == "acc_val":
        value = (result or {}).get("acc_selection")
    if value is None and key == "acc_val":
        value = (milestone or {}).get("acc_selection")
    return value


def _pct(value):
    return "—" if value is None else f"{100.0 * float(value):.3f}%"


def _points(value):
    return "—" if value is None else f"{float(value):+.3f}"


def _mean_sem(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance / len(values))


def load_wave(root):
    root = Path(root).expanduser().resolve()
    manifest = _json(root / "wave_manifest.json") or {"runs": [], "campaign": root.name}
    rows = []
    for order, spec in enumerate(manifest.get("runs", [])):
        run_id = spec["run_id"]
        result = _json(root / f"{run_id}.json")
        milestone = _last_jsonl(root / f"{run_id}.milestones.jsonl")
        trainlog = _last_jsonl(root / f"{run_id}.trainlog.jsonl")
        epoch = (int((result or {}).get("config", {}).get("train", {}).get("epochs", 30))
                 if result else int((trainlog or {}).get("epoch", -1)) + 1)
        rows.append({
            "order": order, **spec, "result": result,
            "state": "complete" if result else ("training" if trainlog else "pending"),
            "epoch": epoch,
            **{metric: _metric(result, milestone, metric) for metric in METRICS},
            "total_params": (result or {}).get("total_params"),
            "active_ffn_params": (result or {}).get("active_ffn_params"),
        })
    return root, manifest, rows


def normalized_config(config):
    """Remove only the two factors intentionally changed between curve endpoints."""
    normalized = copy.deepcopy(config)
    normalized.pop("run_tag", None)
    normalized.get("train", {}).pop("environment_subset", None)
    # ``router_frozen`` was added after the seed-1/2 full anchors were executed.
    # Its historical absence means the same false/default behavior; preserve true as a
    # material difference so a genuinely frozen-router artifact still fails pairing.
    normalized.setdefault("model", {}).setdefault("router_frozen", False)
    return normalized


def validate_artifacts(manifest, rows, expected_environments=None):
    problems = []
    expected_runs = int(manifest.get("expected_runs", 8))
    if len(rows) != expected_runs:
        problems.append(f"expected {expected_runs} rows, found {len(rows)}")
    if len({row["run_id"] for row in rows}) != len(rows):
        problems.append("run ids are not unique")
    for row in rows:
        result = row["result"]
        if not result:
            continue
        if result.get("run_id") != row["run_id"]:
            problems.append(f"{row['label']}: run-id mismatch")
        if int(result.get("seed", -1)) != int(row["seed"]):
            problems.append(f"{row['label']}: seed mismatch")
        if result.get("selection_split") != "ood_val":
            problems.append(f"{row['label']}: wrong checkpoint split")
        if int(result.get("stage", -1)) != 3 or not result.get("test_evaluated"):
            problems.append(f"{row['label']}: not terminal stage 3")
        if result.get("git_dirty"):
            problems.append(f"{row['label']}: dirty execution worktree")
        if expected_environments is not None:
            actual = result.get("config", {}).get("train", {}).get("environment_subset") or []
            if sorted(map(int, actual)) != sorted(map(int, expected_environments)):
                problems.append(f"{row['label']}: training-environment mismatch")
    return problems


def render_report(quarter_root, full_root=None):
    qroot, qmanifest, qrows = load_wave(quarter_root)
    full_root = full_root or qmanifest.get("full_anchor_root")
    froot, fmanifest, frows = load_wave(full_root) if full_root else (None, {"runs": []}, [])
    problems = validate_artifacts(
        qmanifest, qrows, qmanifest.get("quarter_environment_subset", []))
    problems += ["full anchor: " + value for value in validate_artifacts(fmanifest, frows)]

    qmap = {(row["arm"], int(row["seed"])): row for row in qrows}
    fmap = {(row["arm"], int(row["seed"])): row for row in frows}
    for key, quarter in qmap.items():
        full = fmap.get(key)
        if not quarter.get("result") or not full or not full.get("result"):
            continue
        if normalized_config(quarter["result"]["config"]) != normalized_config(
                full["result"]["config"]):
            problems.append(f"{quarter['label']}: quarter/full resolved-config drift")

    complete = sum(row["state"] == "complete" for row in qrows)
    lines = [f"{qmanifest.get('campaign', qroot.name)} — {complete}/{len(qrows)} new rows complete"]
    lines.append("Protocol: " + ("; ".join(problems) if problems else
                                  "all available artifacts and paired configs pass"))
    lines += [
        "| Scale | Seed | Arm | State | Ep | Train | raw ID* | OOD val | OOD test | Worst test |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scale, rows in (("quarter", qrows), ("full anchor", frows)):
        for row in rows:
            if int(row.get("seed", -1)) not in set(map(int, qmanifest.get("seeds", []))):
                continue
            if row.get("arm") not in ARMS:
                continue
            lines.append(
                f"| {scale} | {row['seed']} | {row['arm']} | {row['state']} | {row['epoch']} | "
                f"{_pct(row['acc_train'])} | {_pct(row['acc_within'])} | {_pct(row['acc_val'])} | "
                f"{_pct(row['acc_heldout'])} | {_pct(row['worst_env_heldout'])} |")

    lines += ["", "Architecture-by-scale interactions (full minus quarter contrast; points):",
              "| Contrast | Seed | OOD test | Worst test | OOD val | raw ID* |",
              "|---|---:|---:|---:|---:|---:|"]
    interactions = defaultdict(list)
    for left, right in CONTRASTS:
        for seed in qmanifest.get("seeds", []):
            qa, qb = qmap.get((left, int(seed))), qmap.get((right, int(seed)))
            fa, fb = fmap.get((left, int(seed))), fmap.get((right, int(seed)))
            values = []
            for metric in ("acc_heldout", "worst_env_heldout", "acc_val", "acc_within"):
                if any(row is None or row.get(metric) is None for row in (qa, qb, fa, fb)):
                    value = None
                else:
                    value = 100.0 * ((fa[metric] - fb[metric]) - (qa[metric] - qb[metric]))
                    interactions[(left, right, metric)].append(value)
                values.append(value)
            lines.append(f"| {left} − {right} | {seed} | "
                         + " | ".join(_points(value) for value in values) + " |")

    lines += ["", "Across-seed interaction mean ± SEM (points):",
              "| Contrast | Metric | n | Mean | SEM |", "|---|---|---:|---:|---:|"]
    for left, right in CONTRASTS:
        for label, metric in (("OOD test", "acc_heldout"),
                              ("Worst test", "worst_env_heldout"),
                              ("OOD val", "acc_val")):
            values = interactions.get((left, right, metric), [])
            mean, sem = _mean_sem(values)
            lines.append(f"| {left} − {right} | {label} | {len(values)} | "
                         f"{_points(mean)} | {_points(sem)} |")
    lines.append("")
    lines.append("* raw ID is not a scale-comparable endpoint until fixed-environment re-evaluation.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("quarter_root")
    parser.add_argument("--full-root")
    args = parser.parse_args()
    print(render_report(args.quarter_root, args.full_root))


if __name__ == "__main__":
    main()
