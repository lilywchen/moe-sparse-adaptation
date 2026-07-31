# SciServer execution contract

Persistent repository:
`/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation`

Persistent results:
`/home/idies/workspace/Storage/lchen5/persistent/RESULTS/moe-sparse-adaptation`

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
export MOE_RESULTS=/home/idies/workspace/Storage/lchen5/persistent/RESULTS/moe-sparse-adaptation
python scripts/sweep_ccas.py --dry-run
python scripts/sweep_ccas.py --gpus 0,1 --max-concurrent 2
```

The sweep is idempotent. Launch inside `tmux` so browser or notebook disconnects do not kill it.

## Scheduled check permissions

The scheduled Codex task may read the SciServer page, terminal output, process list, logs, result
counts, and repository status; summarize progress; and run the read-only aggregator.

It must not enter credentials, expose secrets, replace CUDA/PyTorch, delete results, cancel jobs,
launch a new sweep, force-push, or merge branches. If a run is stalled or a protocol guard fails,
it should report the exact evidence and stop for review.
