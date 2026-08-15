#!/usr/bin/env python
"""Change-driven monitor for one RxRx1 HUVEC recipe run."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.certify_rxrx1_huvec_recipe import format_status


DEFAULT_RESULT_ROOT = Path(
    "/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/"
    "substrate_rxrx1/huvec_systematic_fast_20260814"
)


def _status_path(result_root: Path, run_name: str, model: str) -> Path:
    return result_root / "recipe_certification" / run_name / model / "status.json"


def _read(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        # Atomic replacement should make this rare, but a monitor must never kill training.
        return None


def _signature(payload):
    if payload is None:
        return None
    return (
        payload.get("state"), payload.get("attempt_name"), payload.get("epoch"),
        payload.get("updated_at"), payload.get("message"),
    )


def _print(run_name: str, model: str, payload) -> None:
    print(time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print(f"===== {run_name} / {model} =====", flush=True)
    print(format_status(payload), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--run-name", default="parallel_recipe_1_extend160")
    parser.add_argument("--model", default="vit_tiny")
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("--interval must be at least one second")

    path = _status_path(Path(args.result_root).expanduser().resolve(),
                        args.run_name, args.model)
    previous = object()
    while True:
        payload = _read(path)
        signature = _signature(payload)
        if signature != previous:
            _print(args.run_name, args.model, payload)
            previous = signature
        if args.once or (payload and payload.get("state") in {
                "complete", "failed", "interrupted"}):
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
