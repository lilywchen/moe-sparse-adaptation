#!/usr/bin/env python
"""Print one compact live table for an RxRx1 performance-wave result directory.

The table merges the wave manifest, terminal result JSONs, milestone JSONL files, and legacy
``*.ood_test.json`` sidecars.  It works while jobs are running and after they complete, so there
is no per-checkpoint evaluation or aggregation command sequence to remember.
"""
import argparse
import json
from pathlib import Path


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def _last_jsonl(path):
    try:
        lines = [line for line in Path(path).read_text().splitlines() if line.strip()]
        return json.loads(lines[-1]) if lines else None
    except Exception:
        return None


def _pct(value):
    return "—" if value is None else f"{100.0 * float(value):.3f}%"


def collect_rows(result_root):
    root = Path(result_root).expanduser().resolve()
    manifest = _read_json(root / "wave_manifest.json") or {}
    planned = manifest.get("runs", [])
    if not planned:
        planned = []
        for result_path in sorted(root.glob("*.json")):
            if result_path.name.endswith((".ood_test.json", ".artifact_manifest.json")):
                continue
            payload = _read_json(result_path)
            if payload and payload.get("run_id"):
                planned.append({"label": payload["run_id"], "run_id": payload["run_id"]})

    rows = []
    for order, spec in enumerate(planned):
        run_id = spec["run_id"]
        result = _read_json(root / f"{run_id}.json")
        milestone = _last_jsonl(root / f"{run_id}.milestones.jsonl")
        trainlog = _last_jsonl(root / f"{run_id}.trainlog.jsonl")
        sidecar = _read_json(root / f"{run_id}.epoch030.ood_test.json")
        if sidecar is None:
            candidates = sorted(root.glob(f"{run_id}*.ood_test.json"))
            sidecar = _read_json(candidates[-1]) if candidates else None

        if result is not None:
            state = "complete"
            epoch = int(result.get("config", {}).get("train", {}).get("epochs", 30))
        elif trainlog is not None:
            state = "training"
            epoch = int(trainlog.get("epoch", -1)) + 1
        else:
            state, epoch = "pending", 0

        val = (result or {}).get("acc_selection")
        if val is None and milestone is not None:
            val = milestone.get("acc_selection")
        test = (result or {}).get("acc_heldout")
        worst_test = (result or {}).get("worst_env_heldout")
        if sidecar:
            test = sidecar.get("acc_heldout", test)
            worst_test = sidecar.get("worst_env_heldout", worst_test)
        within = (result or {}).get("acc_within")
        if within is None and milestone is not None:
            within = milestone.get("acc_within")
        train = (result or {}).get("acc_train")
        if train is None and milestone is not None:
            train = milestone.get("acc_train")
        rows.append({
            "order": order, "arm": spec.get("label", run_id), "state": state,
            "epoch": epoch, "val": val, "test": test, "worst_test": worst_test,
            "within": within, "train": train,
        })
    return manifest, rows


def render_table(result_root):
    manifest, rows = collect_rows(result_root)
    title = manifest.get("campaign", Path(result_root).name)
    lines = [f"{title} — {sum(row['state'] == 'complete' for row in rows)}/{len(rows)} complete"]
    headers = ("Arm", "State", "Epoch", "OOD val", "OOD test", "Worst test", "ID", "Train")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in sorted(rows, key=lambda item: item["order"]):
        lines.append("| " + " | ".join((
            row["arm"], row["state"], str(row["epoch"]), _pct(row["val"]),
            _pct(row["test"]), _pct(row["worst_test"]), _pct(row["within"]),
            _pct(row["train"]),
        )) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root")
    args = parser.parse_args()
    print(render_table(args.result_root))


if __name__ == "__main__":
    main()
