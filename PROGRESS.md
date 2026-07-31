# Progress ledger

Last verified: 2026-07-31 on SciServer, commit `fa81281`.

## Protocol state

- [x] Scientific question and three analyses frozen in `PLAN.md`.
- [x] 36-cell factorial encoded.
- [x] Exact function-preserving dense initialization implemented.
- [x] SciServer Python tests pass (38/38).
- [x] Local paper build passes (4-page draft).
- [x] Private GitHub remote created and `main` pushed.
- [x] Overleaf project connected to `lilywchen/moe-sparse-adaptation`; `paper/main.tex` compiles.
- [x] SciServer repository connected in persistent storage and environment smoke-tested.
- [x] Real RxRx1 token/cosine MoE forward+backward smoke test passed on H100.
- [x] Real Camelyon17 image/linear MoE forward+backward smoke test passed on H100.
- [x] W&B persistent authentication and live logging verified.
- [x] Private HF results dataset created; read authentication works.
- [ ] HF token write scope corrected and upload verified.
- [ ] Shared hyperparameters selected without OOD-test access.
- [ ] Stage 1 launched.
- [x] Hourly autonomous research steward created; execution contract is in `STEWARD.md`.

## Latest run status

Stage-0 Phase A launched: six shared full-fine-tuning recipe candidates per dataset, seed 0,
selected on OOD validation only. The 12 candidates are split over four 2xH100 containers:

- container 2875: RxRx1 shard 0/2
- container 2874: RxRx1 shard 1/2
- container 2862: Camelyon17 shard 0/2
- container 2859: Camelyon17 shard 1/2

Each container runs two candidates concurrently and queues its third. The image does not provide
`tmux`; launchers use `nohup`, persistent PID files, and persistent logs under `logs/`. W&B showed
the eight active training runs immediately after launch.

## Next safe action

Monitor Phase A for failures and completed JSON files. Select the top two recipes per dataset using
only OOD validation, rerun those candidates on seeds 1 and 2, then freeze the shared recipe before
launching the 36-cell Stage-1 factorial. Do not use OOD test results for this selection.
