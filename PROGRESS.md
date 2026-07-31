# Progress ledger

Last verified: 2026-07-31 17:35 EDT on SciServer, commit `26ad7fa`.

Research-state synchronization: GitHub commit `a2733da` was pulled into the linked Overleaf
project on 2026-07-31; `paper/main.tex` compiled successfully to four pages with no fatal alert.

## Protocol state

- [x] Scientific question and three analyses frozen in `PLAN.md`.
- [x] 36-cell factorial encoded.
- [x] Exact function-preserving dense initialization implemented.
- [x] SciServer Python tests pass (72/72) at the common execution commit.
- [x] Local paper build passes (4-page draft).
- [x] Private GitHub remote created and `main` pushed.
- [x] Overleaf project connected to `lilywchen/moe-sparse-adaptation`; `paper/main.tex` compiles.
- [x] SciServer repository connected in persistent storage and environment smoke-tested.
- [x] Real RxRx1 token/cosine MoE forward+backward smoke test passed on H100.
- [x] Real Camelyon17 image/linear MoE forward+backward smoke test passed on H100.
- [x] W&B persistent authentication and live logging verified.
- [x] Private HF results dataset created; read authentication works.
- [x] HF token has repository-scoped write access; SciServer identity/upload checks pass and all
  eight completed Phase-A JSONs were backfilled to `lilywchen/moe-sparse-adaptation-results`.
- [ ] Shared hyperparameters selected without OOD-test access.
- [ ] Stage 1 launched.
- [x] 15-minute autonomous research steward created; execution contract is in `STEWARD.md`.

## Latest run status

The first 12 Stage-0 Phase-A files completed, but strict provenance validation found that they span
three tested commits (`03167d1`, `448c215`, and `4795202`). They are retained as diagnostic
evidence and are not eligible for formal ranking. The later code also exposed and repaired a raw
environment-ID bookkeeping defect that had collapsed OOD per-environment metrics.

A clean revalidation of the same frozen six full-fine-tuning candidates per dataset is now running
under the single tested commit `26ad7fa`. No scientific factor, factor level, primary outcome, or
selection rule changed. The distinct persistent result root is
`hpo_revalidation_26ad7fa/`; all 12 candidates were dry-run as pending before launch.

Current allocation:

- container 2874: RxRx1 shard 0/2 on GPUs 0 and 1
- container 2862: RxRx1 shard 1/2 on GPUs 0 and 1
- container 2859: Camelyon17 shard 0/2 on GPUs 0 and 1
- container 2875: Camelyon17 shard 1/2 on GPU 1; GPU 0 continues the independent, healthy
  predeclared RxRx1 epoch-budget sanity probe

At launch verification, all eight H100s were at 98--100% utilization: seven revalidation workers
plus the preserved RxRx1 probe. Training logs were advancing and all new runs had live W&B URLs.
GitHub `main` is synchronized to `26ad7fa`. The frozen 36-cell Stage-1 design remains unchanged;
the unreviewed 24-cell alternative from the exploratory SciServer branch was deliberately excluded.

## Next safe action

Validate all 12 common-commit JSONs for schema, finite metrics, exact config identity,
`selection_split=ood_val`, `test_evaluated=false`, and commit `26ad7fa`. Rank within each dataset by
`acc_selection`, breaking ties with `worst_env_val`, then launch seeds 1 and 2 for the top two
recipes per dataset. Continue the RxRx1 sanity diagnosis in parallel; do not use OOD test results
or change the frozen factorial for selection.
