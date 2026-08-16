#!/usr/bin/env python
"""Three-container, six-GPU launcher for the HUVEC residual wave."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.aggregate_rxrx1_huvec_residual_wave import aggregate, collect
from scripts.prepare_rxrx1_huvec_residual_wave import DEFAULT_ANCHOR, DEFAULT_RESULT, prepare
from scripts.analyze_rxrx1_huvec_task_shift import analyze
from scripts.sweep_rxrx1_huvec_batch_effect import run_tasks, wait_for, wait_for_wave


def main():
    p=argparse.ArgumentParser(); p.add_argument("--anchor-root", default=str(DEFAULT_ANCHOR)); p.add_argument("--result-root", default=str(DEFAULT_RESULT)); p.add_argument("--shard-index", type=int, required=True); p.add_argument("--num-shards", type=int, default=3); p.add_argument("--gpus", default="0,1"); p.add_argument("--status", action="store_true"); p.add_argument("--dry-run", action="store_true")
    a=p.parse_args(); root=Path(a.result_root).resolve()
    if a.status:
        table,_=collect(root); print(table.to_string(index=False)); return
    if not 0 <= a.shard_index < a.num_shards: p.error("invalid shard")
    if a.shard_index == 0 and not (root/"wave_manifest.json").is_file(): prepare(a.anchor_root, root)
    wait_for(root/"wave_manifest.json", "frozen 16-run residual manifest")
    manifest=json.loads((root/"wave_manifest.json").read_text())
    if a.shard_index == 0:
        analyze(a.anchor_root)
    local=[row for i,row in enumerate(manifest["runs"]) if i % a.num_shards == a.shard_index]
    print(f"[plan] shard={a.shard_index}/{a.num_shards} local={len(local)} global=16", flush=True)
    if a.dry_run:
        print("\n".join(row["run_id"] for row in local)); return
    run_tasks(local, root, [x.strip() for x in a.gpus.split(",")], 2)
    wait_for_wave(root, manifest, timeout_hours=20)
    report=root/"analysis"/"REPORT.html"
    if a.shard_index == 0 and not report.is_file(): aggregate(root)
    wait_for(report, "residual mechanism report", timeout_hours=1)


if __name__ == "__main__": main()
