#!/usr/bin/env python3
"""One-command status and terminal table for the RxRx3-core competence pilot."""

import json
from pathlib import Path


METRICS = ("acc_train", "acc_within", "acc_val", "acc_heldout", "worst_env_heldout")


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


def load(result_root):
    root = Path(result_root).expanduser().resolve()
    manifest = _json(root / "wave_manifest.json") or {"runs": [], "campaign": root.name}
    rows = []
    for order, spec in enumerate(manifest.get("runs", [])):
        run_id = spec["run_id"]
        result = _json(root / f"{run_id}.json")
        milestone = _last_jsonl(root / f"{run_id}.milestones.jsonl")
        trainlog = _last_jsonl(root / f"{run_id}.trainlog.jsonl")
        epoch = (int((result or {}).get("config", {}).get("train", {}).get("epochs", 10))
                 if result else int((trainlog or {}).get("epoch", -1)) + 1)
        rows.append({
            "order": order, **spec, "result": result,
            "state": "complete" if result else ("training" if trainlog else "pending"),
            "epoch": epoch,
            **{metric: _metric(result, milestone, metric) for metric in METRICS},
        })
    return root, manifest, rows


def _normalized_config(config):
    value = json.loads(json.dumps(config))
    value.pop("sites", None)  # loader-derived K and cell count
    return value


def validate(manifest, rows):
    problems = []
    if len(rows) != 8:
        problems.append(f"expected 8 declared rows, found {len(rows)}")
    if len({row["run_id"] for row in rows}) != len(rows):
        problems.append("run ids are not unique")
    if not (manifest.get("dataset_audit") or {}).get("passed"):
        problems.append("dataset gate is not recorded as passed")
    expected_sha = str(manifest.get("source_git_commit", ""))
    capacity = manifest.get("compute_accounting") or {}
    for row in rows:
        result = row["result"]
        if not result:
            continue
        label = row["label"]
        if result.get("dataset") != "rxrx3_core":
            problems.append(f"{label}: wrong dataset")
        if int(result.get("seed", -1)) != int(row["seed"]):
            problems.append(f"{label}: seed mismatch")
        if int(result.get("stage", -1)) != 3 or not result.get("test_evaluated"):
            problems.append(f"{label}: not a terminal stage-3 readout")
        if result.get("selection_split") != "id_val":
            problems.append(f"{label}: selection split is not frozen id_val")
        if result.get("git_dirty"):
            problems.append(f"{label}: dirty execution worktree")
        if expected_sha and not expected_sha.startswith(str(result.get("git_sha", ""))):
            problems.append(f"{label}: source SHA mismatch")
        if _normalized_config(result.get("config", {})) != _normalized_config(
                row.get("resolved_config", {})):
            problems.append(f"{label}: resolved config drift")
        expected_capacity = capacity.get(row["arm"], {})
        for key in ("total_params", "active_ffn_params"):
            if expected_capacity.get(key) is not None and int(result.get(key, -1)) != int(
                    expected_capacity[key]):
                problems.append(f"{label}: {key} mismatch")
        if result.get("per_env_heldout") and len(result["per_env_heldout"]) != 85:
            problems.append(f"{label}: expected 85 OOD experiment metrics")
        if result.get("per_env_n_heldout") and sum(map(
                int, result["per_env_n_heldout"].values())) != 23855:
            problems.append(f"{label}: OOD denominator is not 23,855")
    return problems


def render_report(result_root):
    root, manifest, rows = load(result_root)
    problems = validate(manifest, rows)
    complete = sum(row["state"] == "complete" for row in rows)
    lines = [
        f"{manifest.get('campaign', root.name)} — {complete}/{len(rows)} complete",
        "Protocol: " + ("; ".join(problems) if problems else "manifest/source/config/artifact gates pass"),
        "",
        "| Seed | Arm | State | Ep | Train | ID val | OOD test | Worst OOD experiment |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['arm']} | {row['state']} | {row['epoch']} | "
            f"{_pct(row['acc_train'])} | {_pct(row['acc_within'])} | "
            f"{_pct(row['acc_heldout'])} | {_pct(row['worst_env_heldout'])} |"
        )
    completed = [row for row in rows if row["result"]]
    by_arm = {}
    for row in completed:
        by_arm.setdefault(row["arm"], []).append(row)
    lines += ["", "Mean terminal readouts across completed seeds:",
              "| Arm | Seeds | Train | ID val | OOD test | Worst OOD experiment |",
              "|---|---:|---:|---:|---:|---:|"]
    for arm in ("original", "dense_E4_late2", "replace_E4k2_late2", "shared_E3k1_late2"):
        values = by_arm.get(arm, [])
        means = []
        for metric in ("acc_train", "acc_within", "acc_heldout", "worst_env_heldout"):
            observed = [row[metric] for row in values if row[metric] is not None]
            means.append(None if not observed else sum(observed) / len(observed))
        lines.append(
            f"| {arm} | {len(values)} | " + " | ".join(_pct(value) for value in means) + " |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root")
    args = parser.parse_args()
    print(render_report(args.result_root))

