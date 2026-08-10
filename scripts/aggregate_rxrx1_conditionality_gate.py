#!/usr/bin/env python3
"""One-command status, paired domain deltas, and uncertainty for the conditionality gate."""
from __future__ import annotations
import json
import math
import random
from collections import defaultdict
from pathlib import Path

ARMS = ("shared_E1_unconditional", "shared_E3_selected", "shared_E3_fullST",
        "dense_E2_active_matched")
CONTRASTS = (
    ("shared_E3_fullST", "shared_E3_selected", "estimator"),
    ("shared_E3_fullST", "shared_E1_unconditional", "conditionality-fullST"),
    ("shared_E3_selected", "shared_E1_unconditional", "conditionality-selected"),
    ("shared_E3_fullST", "dense_E2_active_matched", "conditional-vs-dense"),
)

def _json(path):
    try: return json.loads(Path(path).read_text())
    except Exception: return None

def _last_jsonl(path):
    try:
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        return rows[-1] if rows else None
    except Exception: return None

def _pct(value): return "—" if value is None else f"{100.0 * float(value):.3f}%"
def _pp(value): return "—" if value is None else f"{100.0 * float(value):+.3f}"

def _metric(result, milestone, key):
    value = (result or {}).get(key)
    if value is None: value = (milestone or {}).get(key)
    if value is None and key == "acc_val": value = (result or milestone or {}).get("acc_selection")
    return value

def _env_map(result, key):
    return {str(k): float(v) for k, v in ((result or {}).get(key) or {}).items()}

def _worst_decile(result):
    values = sorted(_env_map(result, "per_env_heldout").values())
    if not values: return None
    tail = values[:max(1, math.ceil(0.1 * len(values)))]
    return sum(tail) / len(tail)

def _weighted_mean(values, weights):
    denominator = sum(weights)
    if denominator <= 0: raise ValueError("paired bootstrap weights must sum to > 0")
    return sum(value * weight for value, weight in zip(values, weights)) / denominator

def _quantile(values, probability):
    ordered = sorted(values)
    if not ordered: raise ValueError("cannot take a quantile of an empty sample")
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper: return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

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
        rows.append({"order": order, **spec, "result": result,
                     "state": "complete" if result else ("training" if trainlog else "pending"),
                     "epoch": epoch,
                     **{key: _metric(result, milestone, key) for key in
                        ("acc_train", "acc_within", "acc_val", "acc_heldout",
                         "worst_env_heldout")},
                     "worst_decile_heldout": _worst_decile(result)})
    return root, manifest, rows

def _normalized_config(config):
    value = json.loads(json.dumps(config)); value.pop("sites", None); return value

def validate(manifest, rows):
    problems, denominators, environment_sets = [], set(), set()
    expected = int(manifest.get("expected_runs", 12))
    if len(rows) != expected: problems.append(f"expected {expected} declared rows, found {len(rows)}")
    if len({row["run_id"] for row in rows}) != len(rows): problems.append("run ids are not unique")
    expected_sha = str(manifest.get("source_git_commit", ""))
    for row in rows:
        result = row["result"]
        if not result: continue
        label = row["label"]
        if result.get("dataset") != "rxrx1": problems.append(f"{label}: wrong dataset")
        if int(result.get("seed", -1)) != int(row["seed"]): problems.append(f"{label}: seed mismatch")
        if int(result.get("stage", -1)) != 3 or not result.get("test_evaluated"):
            problems.append(f"{label}: not terminal stage-3")
        if result.get("selection_split") != "ood_val": problems.append(f"{label}: wrong selection split")
        if result.get("git_dirty"): problems.append(f"{label}: dirty execution worktree")
        if expected_sha and not expected_sha.startswith(str(result.get("git_sha", ""))):
            problems.append(f"{label}: source SHA mismatch")
        if _normalized_config(result.get("config", {})) != _normalized_config(row["resolved_config"]):
            problems.append(f"{label}: resolved config drift")
        protocol, expected_rng = result.get("protocol") or {}, row["resolved_config"]["train"]
        rng = protocol.get("rng_streams") or {}
        for key in ("model_seed", "data_seed", "training_seed"):
            if int(rng.get(key, -1)) != int(expected_rng[key]): problems.append(f"{label}: {key} mismatch")
        if row["arm"] == "shared_E3_fullST":
            if result.get("routing_estimator") != "full_st": problems.append(f"{label}: estimator mismatch")
            if protocol.get("training_all_routed_experts_active") is not True:
                problems.append(f"{label}: dense-training surrogate missing")
            if int(protocol.get("inference_sparse_top_k", -1)) != 1:
                problems.append(f"{label}: evaluation is not top-1")
        counts = _env_map(result, "per_env_n_heldout")
        if counts:
            denominators.add(int(sum(counts.values()))); environment_sets.add(tuple(sorted(counts)))
    if len(denominators) > 1: problems.append(f"OOD denominators drift: {sorted(denominators)}")
    if len(environment_sets) > 1: problems.append("OOD experiment keys drift")
    return problems

def _paired_domain_bootstrap(left_rows, right_rows, draws=20000, seed=20260810):
    left = {int(r["seed"]): r["result"] for r in left_rows if r["result"]}
    right = {int(r["seed"]): r["result"] for r in right_rows if r["result"]}
    seeds = sorted(set(left) & set(right))
    if len(seeds) < 2: return None
    by_seed = {}
    for s in seeds:
        la, ra = _env_map(left[s], "per_env_heldout"), _env_map(right[s], "per_env_heldout")
        lc, rc = _env_map(left[s], "per_env_n_heldout"), _env_map(right[s], "per_env_n_heldout")
        if set(la) != set(ra) or lc != rc: raise ValueError(f"seed {s}: paired support differs")
        names = sorted(la)
        by_seed[s] = ([la[n]-ra[n] for n in names], [lc[n] for n in names])
    observed = sum(_weighted_mean(delta, count) for delta, count in by_seed.values()) / len(seeds)
    rng, samples = random.Random(seed), []
    for _ in range(draws):
        values = []
        for _ in seeds:
            sampled_seed = rng.choice(seeds)
            delta, count = by_seed[sampled_seed]
            picked = [rng.randrange(len(delta)) for _ in delta]
            values.append(_weighted_mean([delta[i] for i in picked], [count[i] for i in picked]))
        samples.append(sum(values) / len(values))
    return {"estimate": observed, "ci95": [_quantile(samples, .025),
                                             _quantile(samples, .975)]}

def render_report(result_root):
    root, manifest, rows = load(result_root); problems = validate(manifest, rows)
    complete = sum(r["state"] == "complete" for r in rows)
    lines = [f"{manifest.get('campaign', root.name)} — {complete}/{len(rows)} complete",
             "Protocol: " + ("; ".join(problems) if problems else
                              "manifest/source/config/artifact/RNG gates pass"), "",
             "| Seed | Arm | State | Ep | Train | ID | OOD val | OOD test | Worst | Worst decile |",
             "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['seed']} | {r['arm']} | {r['state']} | {r['epoch']} | "
                     f"{_pct(r['acc_train'])} | {_pct(r['acc_within'])} | {_pct(r['acc_val'])} | "
                     f"{_pct(r['acc_heldout'])} | {_pct(r['worst_env_heldout'])} | "
                     f"{_pct(r['worst_decile_heldout'])} |")
    by_arm = defaultdict(list)
    for r in rows: by_arm[r["arm"]].append(r)
    lines += ["", "Paired OOD-test contrasts (percentage points):",
              "| Contrast | Seed | Overall | Worst decile |", "|---|---:|---:|---:|"]
    for left, right, name in CONTRASTS:
        lp = {int(r["seed"]): r for r in by_arm[left] if r["result"]}
        rp = {int(r["seed"]): r for r in by_arm[right] if r["result"]}
        for s in manifest.get("seeds", []):
            l, r = lp.get(int(s)), rp.get(int(s))
            overall = None if not l or not r else float(l["acc_heldout"])-float(r["acc_heldout"])
            tail = None if not l or not r else float(l["worst_decile_heldout"])-float(r["worst_decile_heldout"])
            lines.append(f"| {name}: {left} − {right} | {s} | {_pp(overall)} | {_pp(tail)} |")
    if complete == len(rows) and not problems:
        lines += ["", "Hierarchical paired seed + OOD-experiment bootstrap:",
                  "| Contrast | Estimate (pp) | 95% CI (pp) |", "|---|---:|---:|"]
        for left, right, name in CONTRASTS:
            summary = _paired_domain_bootstrap(by_arm[left], by_arm[right]); low, high = summary["ci95"]
            lines.append(f"| {name} | {100*summary['estimate']:+.3f} | "
                         f"[{100*low:+.3f}, {100*high:+.3f}] |")
    return "\n".join(lines)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("result_root")
    print(render_report(parser.parse_args().result_root))
