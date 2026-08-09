#!/usr/bin/env python
"""One-command status and paired summary for the shared-residual confirmation wave."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


METRICS = ("acc_train", "acc_within", "acc_val", "acc_heldout", "worst_env_heldout")
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


def _pct(value):
    return "—" if value is None else f"{100.0 * float(value):.3f}%"


def _num(value):
    return "—" if value is None else f"{float(value):.4f}"


def _metric(result, milestone, key):
    aliases = {"acc_val": "acc_selection"}
    value = (result or {}).get(key)
    if value is None:
        value = (milestone or {}).get(key)
    if value is None and key in aliases:
        value = (result or {}).get(aliases[key])
    if value is None and key in aliases:
        value = (milestone or {}).get(aliases[key])
    return value


def load(root):
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
            "route_reliance": (result or {}).get("route_reliance"),
            "experts_used": (result or {}).get("experts_used"),
            "total_params": (result or {}).get("total_params"),
            "active_ffn_params": (result or {}).get("active_ffn_params"),
        })
    return root, manifest, rows


def validate(manifest, rows):
    problems = []
    if len(rows) != 8:
        problems.append(f"expected 8 declared rows, found {len(rows)}")
    if len({row["run_id"] for row in rows}) != len(rows):
        problems.append("run ids are not unique")
    for row in rows:
        result = row["result"]
        if not result:
            continue
        if int(result.get("seed", -1)) != int(row["seed"]):
            problems.append(f"{row['label']}: seed mismatch")
        if result.get("selection_split") != "ood_val":
            problems.append(f"{row['label']}: selection split is not ood_val")
        if int(result.get("stage", -1)) != 3 or not result.get("test_evaluated"):
            problems.append(f"{row['label']}: not a terminal stage-3 readout")
        if result.get("git_dirty"):
            problems.append(f"{row['label']}: execution worktree was dirty")
        smoothing = result.get("config", {}).get("train", {}).get("label_smoothing", 0.0)
        if float(smoothing) != float(manifest.get("label_smoothing_fixed", 0.0)):
            problems.append(f"{row['label']}: label smoothing mismatch")
    return problems


def _mean_sem(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance / len(values))


def render_report(result_root):
    root, manifest, rows = load(result_root)
    complete = sum(row["state"] == "complete" for row in rows)
    lines = [f"{manifest.get('campaign', root.name)} — {complete}/{len(rows)} complete"]
    problems = validate(manifest, rows)
    if problems:
        lines.append("Protocol: " + "; ".join(problems))
    else:
        lines.append("Protocol: all completed artifacts pass the declared seed/split/stage checks")

    lines += [
        "| Arm | State | Ep | Train | ID | OOD val | OOD test | Worst test | Reliance |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['state']} | {row['epoch']} | "
            f"{_pct(row['acc_train'])} | {_pct(row['acc_within'])} | {_pct(row['acc_val'])} | "
            f"{_pct(row['acc_heldout'])} | {_pct(row['worst_env_heldout'])} | "
            f"{_num(row['route_reliance'])} |")

    by_key = {(row["arm"], int(row["seed"])): row for row in rows if row["result"]}
    lines += ["", "Paired contrasts (absolute percentage points; OOD validation decides):",
              "| Contrast | Seed | d OOD val | d OOD test | d Worst | d ID |",
              "|---|---:|---:|---:|---:|---:|"]
    for left, right in CONTRASTS:
        for seed in manifest.get("seeds", []):
            a, b = by_key.get((left, int(seed))), by_key.get((right, int(seed)))
            vals = []
            for metric in ("acc_val", "acc_heldout", "worst_env_heldout", "acc_within"):
                vals.append(None if not a or not b or a[metric] is None or b[metric] is None
                            else 100.0 * (float(a[metric]) - float(b[metric])))
            lines.append(
                f"| {left} − {right} | {seed} | "
                + " | ".join("—" if value is None else f"{value:+.3f}" for value in vals)
                + " |")

    lines += ["", "Across-seed OOD-validation summary:",
              "| Arm | n | Mean | SEM |", "|---|---:|---:|---:|"]
    grouped = defaultdict(list)
    for row in rows:
        if row["acc_val"] is not None:
            grouped[row["arm"]].append(row["acc_val"])
    for arm in ("original", "dense_E4_late2", "replace_E4k2_late2", "shared_E3k1_late2"):
        mean, sem = _mean_sem(grouped.get(arm, []))
        lines.append(f"| {arm} | {len(grouped.get(arm, []))} | {_pct(mean)} | {_pct(sem)} |")

    complete_rows = [row for row in rows if row["result"]]
    expanded = [row for row in complete_rows if row["arm"] != "original"]
    if expanded:
        totals = {row["arm"]: row["total_params"] for row in expanded}
        active = {row["arm"]: row["active_ffn_params"] for row in expanded}
        lines.append("")
        lines.append("Capacity audit (first completed row per arm):")
        for arm in ("dense_E4_late2", "replace_E4k2_late2", "shared_E3k1_late2"):
            if totals.get(arm) is not None:
                lines.append(f"  {arm}: total={int(totals[arm]):,}, "
                             f"active-FFN={int(active[arm]):,}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root")
    args = parser.parse_args()
    print(render_report(args.result_root))


if __name__ == "__main__":
    main()
