#!/usr/bin/env python
"""One-command status, terminal table, and architecture-by-scale contrasts."""
import json
from pathlib import Path


METRICS = ("acc_train", "acc_within", "acc_val", "acc_heldout", "worst_env_heldout")
ARMS = ("original", "dense_E4_late2", "replace_E4k2_late2", "shared_E3k1_late2")
SCALES = ("quarter", "full")


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
        value = (result or milestone or {}).get("acc_selection")
    return value


def _pct(value):
    return "—" if value is None else f"{100.0 * float(value):.3f}%"


def _points(left, right):
    return "—" if left is None or right is None else f"{100.0 * (left - right):+.3f}"


def load(result_root):
    root = Path(result_root).expanduser().resolve()
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
        })
    return root, manifest, rows


def validate(manifest, rows):
    problems = []
    if len(rows) != 8:
        problems.append(f"expected 8 declared rows, found {len(rows)}")
    if len({row["run_id"] for row in rows}) != len(rows):
        problems.append("run ids are not unique")
    audit = manifest.get("dataset_audit") or {}
    if audit:
        q, f = audit.get("quarter", {}), audit.get("full", {})
        if q.get("n_classes_observed") != f.get("n_classes_observed"):
            problems.append("quarter/full label coverage differs")
        if set(map(int, q.get("cell_environment_counts", {}))) != {0, 1, 2, 3}:
            problems.append("quarter subset does not cover all four cell types")
    for row in rows:
        result = row["result"]
        if not result:
            continue
        if int(result.get("seed", -1)) != int(row["seed"]):
            problems.append(f"{row['label']}: seed mismatch")
        if int(result.get("stage", -1)) != 3 or not result.get("test_evaluated"):
            problems.append(f"{row['label']}: not a terminal stage-3 readout")
        if result.get("git_dirty"):
            problems.append(f"{row['label']}: dirty execution worktree")
        actual = (result.get("config", {}).get("train", {}).get("environment_subset") or [])
        expected = row.get("environment_subset") or []
        if sorted(map(int, actual)) != sorted(map(int, expected)):
            problems.append(f"{row['label']}: environment subset mismatch")
    return problems


def render_report(result_root):
    root, manifest, rows = load(result_root)
    complete = sum(row["state"] == "complete" for row in rows)
    problems = validate(manifest, rows)
    lines = [f"{manifest.get('campaign', root.name)} — {complete}/{len(rows)} complete",
             "Protocol: " + ("; ".join(problems) if problems else
                              "declared scale/seed/stage checks pass")]
    audit = manifest.get("dataset_audit") or {}
    if audit:
        lines += ["", "Dataset scale (fixed 1,139-class coverage and fixed OOD evaluation):",
                  "| Scale | Train environments | Fields | Wells | Sites | Min/median/max per class |",
                  "|---|---:|---:|---:|---:|---:|"]
        for scale in SCALES:
            item = audit.get(scale, {})
            lines.append(
                f"| {scale} | {item.get('n_environments', '—')} | {item.get('n_fields', '—')} | "
                f"{item.get('n_wells', '—')} | {item.get('n_sites', '—')} | "
                f"{item.get('examples_per_class_min', '—')}/"
                f"{item.get('examples_per_class_median', '—')}/"
                f"{item.get('examples_per_class_max', '—')} |")
    lines += ["", "Headline terminal readouts:",
              "| Scale | Arm | State | Ep | Train | ID | OOD val | OOD test | Worst test |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['scale']} | {row['arm']} | {row['state']} | {row['epoch']} | "
            f"{_pct(row['acc_train'])} | {_pct(row['acc_within'])} | {_pct(row['acc_val'])} | "
            f"{_pct(row['acc_heldout'])} | {_pct(row['worst_env_heldout'])} |")

    by_key = {(row["scale"], row["arm"]): row for row in rows if row["result"]}
    lines += ["", "Paired architecture contrasts (percentage points; test is the headline):",
              "| Scale | Contrast | d Test | d Worst | d ID | d Val |",
              "|---|---|---:|---:|---:|---:|"]
    contrasts = (("shared_E3k1_late2", "dense_E4_late2"),
                 ("shared_E3k1_late2", "replace_E4k2_late2"),
                 ("dense_E4_late2", "original"))
    for scale in SCALES:
        for left_name, right_name in contrasts:
            left, right = by_key.get((scale, left_name)), by_key.get((scale, right_name))
            values = []
            for metric in ("acc_heldout", "worst_env_heldout", "acc_within", "acc_val"):
                values.append(_points(None if left is None else left[metric],
                                      None if right is None else right[metric]))
            lines.append(f"| {scale} | {left_name} − {right_name} | " +
                         " | ".join(values) + " |")

    quarter_shared = by_key.get(("quarter", "shared_E3k1_late2"))
    quarter_dense = by_key.get(("quarter", "dense_E4_late2"))
    full_shared = by_key.get(("full", "shared_E3k1_late2"))
    full_dense = by_key.get(("full", "dense_E4_late2"))
    lines += ["", "Architecture × scale interaction: (shared − dense at full) − "
              "(shared − dense at quarter)",
              "| Metric | Interaction (points) |", "|---|---:|"]
    for label, metric in (("OOD test", "acc_heldout"), ("Worst test", "worst_env_heldout"),
                          ("ID", "acc_within"), ("OOD val", "acc_val")):
        if None in (quarter_shared, quarter_dense, full_shared, full_dense):
            value = "—"
        else:
            value = f"{100.0 * ((full_shared[metric] - full_dense[metric]) - (quarter_shared[metric] - quarter_dense[metric])):+.3f}"
        lines.append(f"| {label} | {value} |")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root")
    args = parser.parse_args()
    print(render_report(args.result_root))
