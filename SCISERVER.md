# SciServer execution contract

Persistent repository:
`/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation`

Persistent results:
`/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/results`

Persistent Stage-0 HPO results:
`/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/hpo`

Anything outside `/home/idies/workspace/Storage/` may disappear with the container.

## First connection checklist

1. Confirm the browser is on `apps.sciserver.org` and already signed in. Never ask an automation
   to type or store a password, token, or recovery code.
2. Open the existing GPU compute container and verify the persistent repository path.
3. Pull `main`, then record the exact commit in `PROGRESS.md`.
4. Verify the container's CUDA-matched PyTorch before installing this package.
5. Install only the Python extras with `python -m pip install -e .`; do not replace torch or
   torchvision unless they are absent and the CUDA version has been checked.
6. Run `pytest -q` and the one-seed dry run. Do not start the factorial if either fails.

## Launch

```bash
export MOE_RESULTS=/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/results
python scripts/sweep_ccas.py --dry-run
python scripts/sweep_ccas.py --gpus 0,1 --max-concurrent 2
```

The sweep is idempotent. Prefer `tmux` when available. The current GPU image does not include it,
so launch with `nohup`, redirect to the persistent `logs/` directory, and save the launcher PID
there. Never rely on a browser terminal remaining connected.

## Scheduled research steward

The scheduled Codex task is an executor governed by `STEWARD.md`, not a read-only checker. It may
inspect remote state, diagnose and fix bounded operational defects, add regression tests, update
the progress/analysis/manuscript state, restart one missing idempotent candidate after a verified
transient failure, and launch the next predeclared phase once every stage gate passes.

If the expected job is healthy, it emits only a compact heartbeat and does not relaunch or disturb
it. If the job has finished, it owns the handoff through validation, aggregation, state updates,
and the next valid launch. It must never enter credentials, expose secrets, replace CUDA/PyTorch,
delete results, cancel healthy jobs, touch another project's resources, force-push, alter the
scientific protocol, inspect OOD test early, or retry a repeated failure without review.
