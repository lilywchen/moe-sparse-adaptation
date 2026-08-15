#!/usr/bin/env python
"""Compact, change-driven monitor for the two-container HUVEC batch study."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path


def snapshot(root):
    root = Path(root).expanduser().resolve()
    manifest_path = root / "wave_manifest.json"
    if not manifest_path.is_file():
        return "[pending] waiting for the frozen manifest"
    manifest = json.loads(manifest_path.read_text())
    states, active = {}, []
    for spec in manifest["runs"]:
        run_dir = root / "runs" / spec["run_id"]
        result_path, status_path = run_dir / "RESULT.json", run_dir / "STATUS.json"
        if result_path.is_file():
            state = "complete"; payload = json.loads(result_path.read_text())
        elif status_path.is_file():
            payload = json.loads(status_path.read_text()); state = payload.get("state", "unknown")
        else:
            payload = {}; state = "pending"
        states[state] = states.get(state, 0) + 1
        if state in ("training", "failed", "interrupted"):
            latest = payload.get("latest", {})
            active.append(
                f"{state:11s} gpu-job={spec['run_id']} epoch={payload.get('epoch', 0)} "
                f"train={latest.get('train_augmented_site_top1', float('nan')):.4f} "
                f"IID={payload.get('best_source_iid_site_top1', float('nan')):.4f}")
    lines = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        f"complete={states.get('complete', 0)}/36 training={states.get('training', 0)} "
        f"pending={states.get('pending', 0)} failed={states.get('failed', 0)} "
        f"interrupted={states.get('interrupted', 0)}"]
    lines.extend(active or ["No active run status has been written yet."])
    report = root / "analysis" / "REPORT.html"
    if report.is_file():
        lines.append(f"FINAL REPORT: {report}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args(); previous = None
    while True:
        value = snapshot(args.result_root)
        digest = hashlib.sha256(value.encode()).hexdigest()
        if digest != previous:
            print(value, flush=True); previous = digest
        if not args.watch or "FINAL REPORT:" in value:
            return
        time.sleep(max(5, args.interval))


if __name__ == "__main__":
    main()
