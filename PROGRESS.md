# Progress ledger

Last verified: 2026-07-31 18:12 EDT on SciServer.

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

Current allocation and progress:

- container 2874: RxRx1 shard 0/2 on GPUs 0 and 1
- container 2862: RxRx1 shard 1/2 on GPUs 0 and 1
- container 2859: Camelyon17 shard 0/2 on GPUs 0 and 1
- container 2875: Camelyon17 shard 1/2 on GPU 1; canonical WILDS RxRx1 ERM sanity reproduction
  on GPU 0

All eight H100s are assigned to expected work. The four RxRx1 Phase-A workers reached epoch 29/29
and are in end-of-run evaluation. The three Camelyon17 workers were at epochs 4/10, 2/10, and 1/10
in the latest logs. No common-commit result JSON is complete yet (`0/12`), so no formal ranking is
licensed. All seven revalidation processes use `26ad7fa`; their persistent root is
`hpo_revalidation_26ad7fa/`.

Both 90-epoch RxRx1 DINOv2 probes completed but remain diagnostic: OOD-validation accuracy was
0.0140/0.0177 and seen-environment accuracy was 0.0945/0.1042 for LLRD 0.70/0.85. Their files are
parseable and test-blind but were produced from dirty commit `4795202`, so they cannot support
selection. They establish a substantive sanity concern, not an MoE conclusion.

The idle GPU created by those completed probes was filled with a canonical WILDS ResNet-50 ERM
reproduction. A zero-epoch dry-run verified the official RxRx1 transform and optimizer schedule,
and an added leakage guard restricted construction/evaluation to train, validation, and ID-test.
The first launch failed before meaningful training because the environment lacks optional
`torch_scatter`; the failed output and W&B run are retained as an explicit exclusion. A native
PyTorch arithmetic-mean fallback was unit-smoked, and the single allowed retry is healthy under
compatibility commit `6fb65e5`: process alive, GPU 0 at 95%+ with 11.3 GB allocated, training log
advancing in epoch 0, W&B run `rytuap3l`, and zero fatal matches at verification.

GitHub/local source remains the scientific source of truth. The frozen 36-cell Stage-1 design is
unchanged; the unreviewed 24-cell alternative from the exploratory SciServer branch remains
excluded.

## Next safe action

Continue all eight healthy jobs. As soon as a dataset's six common-commit JSONs exist, validate
schema, finite metrics, exact config identity, `selection_split=ood_val`, `test_evaluated=false`,
and commit `26ad7fa`; then rank that dataset by `acc_selection` with `worst_env_val` as tie-breaker.
For RxRx1, compare the canonical ERM learning regime with the DINOv2 probes before replicating a
potentially invalid recipe. Do not access OOD test, change factorial factors, or treat mechanism
analysis as evidence before the predictive comparison is valid.
