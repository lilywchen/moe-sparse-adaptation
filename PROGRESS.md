# Progress ledger

Last verified: 2026-07-31 18:58 EDT on SciServer.

Research-state synchronization: GitHub commit `5f8310c` was pulled into the linked Overleaf
project on 2026-07-31; `paper/main.tex` compiled successfully to four pages with 0 errors, one
pre-existing package warning, and two nonfatal typesetting notices.

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

- container 2875: canonical WILDS RxRx1 ERM sanity reproduction on GPU 0 and one Camelyon17
  Phase-A candidate on GPU 1
- container 2874: controlled RxRx1 `no random-resized crop` diagnostic on GPU 0; GPU 1 idle
- container 2862: controlled RxRx1 `uniform layer learning rate` diagnostic on GPU 0; GPU 1 idle
- container 2859: two Camelyon17 Phase-A candidates on GPUs 0 and 1

All six RxRx1 Phase-A candidates have now completed and passed strict validation (`6/6` RxRx1).
One Camelyon17 candidate is also valid, giving `7/12` formal Stage-0 results overall. Every valid
file is parseable and finite, has exact filename/config identity, uses the clean common commit
`26ad7fa`, records `selection_split=ood_val` and `test_evaluated=false`, and has the same training
parameter count (21,628,800). The RxRx1 ranking by the frozen rule is `(1e-4, 0.85)`, `(3e-4,
0.85)`, `(1e-4, 0.70)`, `(3e-4, 0.70)`, `(3e-5, 0.85)`, `(3e-5, 0.70)`, but no recipe is frozen
or replicated because the entire grid remains in a clearly inadequate accuracy regime.

Six H100s are active: three Camelyon17 candidates, the canonical RxRx1 ERM control, and two
single-factor RxRx1 substrate diagnostics. Two H100s are idle because no additional diagnostic is
needed before these causal checks return. All formal revalidation work uses `26ad7fa` and the
persistent root `hpo_revalidation_26ad7fa/`. The two new diagnostics also execute clean commit
`26ad7fa` but write to the separate diagnostic root `hpo/rxrx1/dense_rescue_26ad7fa/` and cannot
enter the formal HPO ranking.

Both 90-epoch RxRx1 DINOv2 probes completed but remain diagnostic: OOD-validation accuracy was
0.0140/0.0177 and seen-environment accuracy was 0.0945/0.1042 for LLRD 0.70/0.85. Their files are
parseable and test-blind but were produced from dirty commit `4795202`, so they cannot support
selection. They establish a substantive sanity concern, not an MoE conclusion.

The canonical WILDS ResNet-50 ERM reproduction is now at epoch 22. At the last fully completed
evaluation (epoch 21), it reached 24.6% ID-test accuracy and 13.4% OOD-validation accuracy; its
training-set evaluation was 70.7%. This is decisive diagnostic evidence that the dataset, labels,
and split plumbing are learnable in the current environment. It is not a model-selection baseline
for the MoE factorial and does not use the OOD-test split.

Because the official control learned while every DINOv2 recipe remained weak, two bounded
single-factor diagnostics were launched from the best formal DINOv2 anchor `(lr=1e-4, LLRD=0.85)`:
one removes random-resized cropping while holding optimization fixed, and one sets LLRD to 1.0
while retaining the crop. Both passed config/idempotency/clean-commit/GPU preflight and were
independently verified on GPU 0 of containers 2874 and 2862 at 99% utilization with the expected
command identity, fresh persistent logs, and OOD-test blindness.

GitHub/local source remains the scientific source of truth. The frozen 36-cell Stage-1 design is
unchanged; the unreviewed 24-cell alternative from the exploratory SciServer branch remains
excluded.

## Next safe action

Continue the six healthy jobs. When the two RxRx1 diagnostics finish, strictly validate them and
compare each only with the fixed `(1e-4, 0.85)` anchor. If either produces a large recovery, freeze
the corresponding corrected adaptation recipe and rerun the small shared screen before any MoE
factorial work. If neither approaches a credible dense regime while the canonical ERM remains
strong, stop using DINOv2 as the RxRx1 substrate and make an explicit backbone decision rather
than spending the factorial budget on a floor baseline. Continue and validate Camelyon17 Phase A
independently. Do not access OOD test or launch Stage 1 before these gates resolve.
