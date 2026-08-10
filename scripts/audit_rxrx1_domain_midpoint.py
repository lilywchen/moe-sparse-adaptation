#!/usr/bin/env python
"""Strict terminal artifact audit for the RxRx1 midpoint wave."""
import argparse
import copy
import json
from pathlib import Path


EXPECTED_PARAMS = {
    "dense_E4_late2": (29493881, 9454854),
    "replace_E4k2_late2": (29494645, 4729346),
    "shared_E3k1_late2": (29493877, 4728578),
}


def _json(path):
    return json.loads(Path(path).read_text())


def _last_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return rows[-1] if rows else None


def _declared_config(config):
    value = copy.deepcopy(config)
    value.pop("sites", None)  # Deterministic runtime dataset summary, not a training factor.
    return value


def audit(root):
    root = Path(root).expanduser().resolve()
    manifest = _json(root / "wave_manifest.json")
    problems, rows = [], []
    expected_sha = str(manifest["source_git_commit"])
    expected_envs = sorted(map(int, manifest["midpoint_environment_subset"]))
    if manifest.get("source_git_dirty"):
        problems.append("manifest source is dirty")
    if len(manifest.get("runs", [])) != int(manifest.get("expected_runs", 8)):
        problems.append("manifest row count mismatch")
    for spec in manifest["runs"]:
        run_id = spec["run_id"]
        result_path = root / f"{run_id}.json"
        trainlog_path = root / f"{run_id}.trainlog.jsonl"
        if not result_path.is_file():
            problems.append(f"{spec['label']}: missing result")
            continue
        result = _json(result_path)
        last_train = _last_jsonl(trainlog_path) if trainlog_path.is_file() else None
        checks = {
            "run_id": result.get("run_id") == run_id,
            "seed": int(result.get("seed", -1)) == int(spec["seed"]),
            "stage3": int(result.get("stage", -1)) == 3 and bool(result.get("test_evaluated")),
            "selection": result.get("selection_split") == "ood_val",
            "clean": not bool(result.get("git_dirty")),
            "source_sha": expected_sha.startswith(str(result.get("git_sha", ""))),
            "resolved_config": _declared_config(result.get("config", {})) == _declared_config(spec["resolved_config"]),
            "environment_subset": sorted(map(int, result.get("config", {}).get("train", {}).get("environment_subset", []))) == expected_envs,
            "epoch30": bool(last_train) and int(last_train.get("epoch", -1)) == 29,
            "train_accuracy_recorded": result.get("acc_train") is not None,
        }
        per_env = {str(k): float(v) for k, v in result.get("per_env_heldout", {}).items()}
        counts = {str(k): int(v) for k, v in result.get("per_env_n_heldout", {}).items()}
        checks["batch_metrics"] = bool(per_env) and set(per_env) == set(counts) and all(v > 0 for v in counts.values())
        checks["worst_batch"] = bool(per_env) and abs(float(result.get("worst_env_heldout", -1)) - min(per_env.values())) < 1e-12
        if spec["arm"] in EXPECTED_PARAMS:
            checks["parameter_counts"] = (
                int(result.get("total_params", -1)), int(result.get("active_ffn_params", -1))) == EXPECTED_PARAMS[spec["arm"]]
        for name, passed in checks.items():
            if not passed:
                problems.append(f"{spec['label']}: {name} failed")
        rows.append({"label": spec["label"], "run_id": run_id, "arm": spec["arm"],
                     "seed": spec["seed"], "checks": checks})
    return {"campaign": manifest.get("campaign"), "source_git_commit": expected_sha,
            "expected_runs": manifest.get("expected_runs"), "audited_rows": len(rows),
            "passed": not problems, "problems": problems, "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--output-json")
    args = parser.parse_args()
    report = audit(args.root)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
