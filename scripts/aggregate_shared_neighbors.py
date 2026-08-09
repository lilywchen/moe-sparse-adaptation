#!/usr/bin/env python
"""One-command status and paired summary for shared-residual neighbors."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


METRICS = ("acc_train", "acc_within", "acc_val", "acc_heldout", "worst_env_heldout")
ARMS = (
    "shared_E2k1_late2",
    "shared_E4k1_late2",
    "shared_E3k1_block10",
    "shared_E3k1_block11",
)
ANCHOR_ARM = "shared_E3k1_late2"
CONTRASTS = (
    ("shared_E2k1_late2", ANCHOR_ARM),
    ("shared_E4k1_late2", ANCHOR_ARM),
    ("shared_E3k1_block10", ANCHOR_ARM),
    ("shared_E3k1_block11", ANCHOR_ARM),
    ("shared_E3k1_block11", "shared_E3k1_block10"),
    ("shared_E4k1_late2", "shared_E2k1_late2"),
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
    value = (result or {}).get(key)
    if value is None:
        value = (milestone or {}).get(key)
    if value is None and key == "acc_val":
        value = (result or {}).get("acc_selection")
    if value is None and key == "acc_val":
        value = (milestone or {}).get("acc_selection")
    return value


def _row(root, spec, order):
    run_id = spec["run_id"]
    result = _json(root / f"{run_id}.json")
    milestone = _last_jsonl(root / f"{run_id}.milestones.jsonl")
    trainlog = _last_jsonl(root / f"{run_id}.trainlog.jsonl")
    epoch = (int((result or {}).get("config", {}).get("train", {}).get("epochs", 30))
             if result else int((trainlog or {}).get("epoch", -1)) + 1)
    return {
        "order": order, **spec, "result": result,
        "state": "complete" if result else ("training" if trainlog else "pending"),
        "epoch": epoch,
        **{metric: _metric(result, milestone, metric) for metric in METRICS},
        "route_reliance": (result or {}).get("route_reliance"),
        "total_params": (result or {}).get("total_params"),
        "active_ffn_params": (result or {}).get("active_ffn_params"),
    }


def load(root):
    root = Path(root).expanduser().resolve()
    manifest = _json(root / "wave_manifest.json") or {"runs": [], "campaign": root.name}
    rows = [_row(root, spec, order) for order, spec in enumerate(manifest.get("runs", []))]
    return root, manifest, rows


def _anchors(manifest):
    root = Path(manifest.get("anchor_root", "")).expanduser()
    anchor_manifest = _json(root / "wave_manifest.json") or {"runs": []}
    rows = []
    for spec in anchor_manifest.get("runs", []):
        if spec.get("arm") == ANCHOR_ARM and int(spec.get("seed", -1)) in manifest.get("seeds", []):
            rows.append(_row(root, spec, len(rows)))
    return rows


def validate(manifest, rows, anchors):
    problems = []
    if len(rows) != 8:
        problems.append(f"expected 8 declared rows, found {len(rows)}")
    if len({row["run_id"] for row in rows}) != len(rows):
        problems.append("run ids are not unique")
    if len(anchors) != len(manifest.get("seeds", [])):
        problems.append("same-seed shared-E3 late2 anchors are missing")
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
    anchors = _anchors(manifest)
    complete = sum(row["state"] == "complete" for row in rows)
    lines = [f"{manifest.get('campaign', root.name)} — {complete}/{len(rows)} complete"]
    problems = validate(manifest, rows, anchors)
    lines.append("Protocol: " + ("; ".join(problems) if problems else
                                  "completed rows and same-seed anchors pass declared checks"))
    lines += [
        "| Arm | State | Ep | Train | ID | OOD val | OOD test | Worst test | Reliance |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['state']} | {row['epoch']} | {_pct(row['acc_train'])} | "
            f"{_pct(row['acc_within'])} | {_pct(row['acc_val'])} | {_pct(row['acc_heldout'])} | "
            f"{_pct(row['worst_env_heldout'])} | {_num(row['route_reliance'])} |")

    by_key = {(row["arm"], int(row["seed"])): row for row in [*rows, *anchors]
              if row["result"]}
    lines += ["", "Paired contrasts (points; OOD validation decides):",
              "| Contrast | Seed | d OOD val | d OOD test | d Worst | d ID |",
              "|---|---:|---:|---:|---:|---:|"]
    for left, right in CONTRASTS:
        for seed in manifest.get("seeds", []):
            a, b = by_key.get((left, int(seed))), by_key.get((right, int(seed)))
            values = []
            for metric in ("acc_val", "acc_heldout", "worst_env_heldout", "acc_within"):
                values.append(None if not a or not b or a[metric] is None or b[metric] is None
                              else 100.0 * (float(a[metric]) - float(b[metric])))
            lines.append(f"| {left} − {right} | {seed} | " + " | ".join(
                "—" if value is None else f"{value:+.3f}" for value in values) + " |")

    grouped = defaultdict(list)
    for row in [*rows, *anchors]:
        if row["acc_val"] is not None:
            grouped[row["arm"]].append(row["acc_val"])
    lines += ["", "Across-seed OOD-validation summary:",
              "| Arm | n | Mean | SEM |", "|---|---:|---:|---:|"]
    for arm in (*ARMS, ANCHOR_ARM):
        mean, sem = _mean_sem(grouped.get(arm, []))
        lines.append(f"| {arm} | {len(grouped.get(arm, []))} | {_pct(mean)} | {_pct(sem)} |")

    complete_rows = [row for row in rows if row["result"]]
    if complete_rows:
        lines += ["", "Capacity audit (first completed row per arm):"]
        for arm in ARMS:
            row = next((item for item in complete_rows if item["arm"] == arm), None)
            if row and row["total_params"] is not None:
                lines.append(f"  {arm}: total={int(row['total_params']):,}, "
                             f"active-FFN={int(row['active_ffn_params']):,}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root")
    args = parser.parse_args()
    print(render_report(args.result_root))


if __name__ == "__main__":
    main()
