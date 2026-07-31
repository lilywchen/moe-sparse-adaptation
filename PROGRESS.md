# Progress ledger

Last verified: 2026-07-31 19:14 EDT on SciServer.

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
- container 2874: RxRx1 `no random-resized crop` on GPU 0 and `no crop + uniform layer LR` on GPU 1
- container 2862: RxRx1 `uniform layer LR` on GPU 0 and `official-style preprocessing + uniform
  layer LR` on GPU 1
- container 2859: two Camelyon17 Phase-A candidates on GPUs 0 and 1

All six RxRx1 Phase-A candidates have now completed and passed strict validation (`6/6` RxRx1).
One Camelyon17 candidate is also valid, giving `7/12` formal Stage-0 results overall. Every valid
file is parseable and finite, has exact filename/config identity, uses the clean common commit
`26ad7fa`, records `selection_split=ood_val` and `test_evaluated=false`, and has the same training
parameter count (21,628,800). The RxRx1 ranking by the frozen rule is `(1e-4, 0.85)`, `(3e-4,
0.85)`, `(1e-4, 0.70)`, `(3e-4, 0.70)`, `(3e-5, 0.85)`, `(3e-5, 0.70)`, but no recipe is frozen
or replicated because the entire grid remains in a clearly inadequate accuracy regime.

All eight H100s are active: three Camelyon17 candidates, the canonical RxRx1 ERM control, and four
bounded RxRx1 substrate diagnostics. All formal revalidation work uses `26ad7fa` and the persistent
root `hpo_revalidation_26ad7fa/`. The no-crop, uniform-LR, and combined diagnostics execute clean
commit `26ad7fa` under `hpo/rxrx1/dense_rescue_26ad7fa/`. The official-style transform is GitHub
commit `aa8d0cf`, applied and regression-tested as clean isolated SciServer commit `1da67a5`, under
`hpo/rxrx1/dense_rescue_1da67a5/`. These are diagnostic and cannot enter the formal HPO ranking.

Both 90-epoch RxRx1 DINOv2 probes completed but remain diagnostic: OOD-validation accuracy was
0.0140/0.0177 and seen-environment accuracy was 0.0945/0.1042 for LLRD 0.70/0.85. Their files are
parseable and test-blind but were produced from dirty commit `4795202`, so they cannot support
selection. They establish a substantive sanity concern, not an MoE conclusion.

The canonical WILDS ResNet-50 ERM reproduction is now at epoch 22. At the last fully completed
evaluation (epoch 21), it reached 24.6% ID-test accuracy and 13.4% OOD-validation accuracy; its
training-set evaluation was 70.7%. This is decisive diagnostic evidence that the dataset, labels,
and split plumbing are learnable in the current environment. It is not a model-selection baseline
for the MoE factorial and does not use the OOD-test split.

Because the official control learned while every DINOv2 recipe remained weak, the bounded rescue
set now contains four predeclared arms from the `(lr=1e-4, LLRD=0.85)` anchor: remove cropping;
set LLRD to 1.0; combine both; and use official WILDS RxRx1 transform semantics with uniform LR.
The latter uses discrete right-angle rotations, horizontal flips, and per-image/per-channel
standardization, plus the deterministic 224 resize required by DINOv2. The transform change passed
3 focused tests and the full SciServer suite (75/75). All four jobs passed clean-checkout,
config/idempotency, persistent-path, test-blindness, and free-GPU checks and were independently
verified at 99% utilization with expected command identities and advancing logs.

The apparent cross-dataset recipe symmetry was incomplete: LR/LLRD candidates were shared, but
RxRx1 used 30 epochs/3 warmup epochs and honored random-resized cropping, while Camelyon17 used 10
epochs/1 warmup epoch and its loader bypassed the crop flag. Dataset-specific recipes are allowed,
but this difference is now explicit: competence-tune once per dataset, freeze it, and apply it
identically to every dense and MoE cell within that dataset.

GitHub/local source remains the scientific source of truth. The frozen 36-cell Stage-1 design is
unchanged; the unreviewed 24-cell alternative from the exploratory SciServer branch remains
excluded.

## Next safe action

Continue all eight healthy jobs. Strictly validate the four rescue JSONs and compare them with the
fixed anchor and the final/best canonical validation-only control. Keep DINOv2 automatically only
if the best rescue reaches at least 80% of canonical OOD-val and ID-test accuracy; abandon it
automatically if it remains below 50% on either metric; ask for one backbone decision in the middle
zone. Do not add rescue arms. If DINOv2 passes, rerun the small RxRx1 dense recipe screen from one
clean frozen commit before any MoE factorial work. Continue Camelyon17 independently. Do not access
OOD test or launch Stage 1 before these gates resolve.

## 2026-07-31 19:38 EDT material update

Two of four bounded RxRx1 rescue arms completed and passed strict validation. Both are clean
`26ad7fa`, use the exact frozen anchor configuration except for their named intervention, contain
finite metrics and exact run/config identity, retain 21,628,800 training parameters, record
`selection_split=ood_val`, and record `test_evaluated=false`.

- `rxdiag_no_rrc`: OOD-val 0.015425, seen-environment 0.045134, worst-environment val 0.001218.
- `rxdiag_uniform_lr`: OOD-val 0.013294, seen-environment 0.036738, worst-environment val 0.000812.

Removing random-resized crop therefore produces only a small improvement over the failed anchor
(0.014816 OOD-val, 0.038043 seen), while uniform layer learning rates alone are worse. Both remain
far below half of the canonical control on both gate metrics and cannot rescue DINOv2 by themselves.
This is diagnostic rather than a final backbone decision because the combined and official-style
preprocessing arms remain unfinished.

The combined no-crop/uniform-LR run completed epoch 29 and is alive in its CPU-heavy validation and
decodability phase; its latest persistent log was 2 minutes old and contained no fatal error. The
official-style transform run reached epoch 23 with loss 4.3387 and a fresh log under clean isolated
commit `1da67a5`; this lower training loss is encouraging but is not interpreted as performance.
Six H100s are active: the combined rescue evaluation, official-style rescue training, three
Camelyon17 workers, and the canonical RxRx1 control. Two H100s are idle because the completed rescue
arms freed them and no additional RxRx1 arm is authorized before the frozen competence decision.

A second Camelyon17 candidate also completed and passed the same formal Stage-0 validation checks,
raising common-commit coverage to 8/12 overall and Camelyon17 coverage to 2/6. The new candidate
`camelyon17_original_ep10_s0_hpoA_lr3e-05_llrd0.85` reached OOD-val 0.885601 and seen-environment
accuracy 0.994666. This is promising but the incomplete six-cell grid cannot yet support selection.

Next: validate the combined and official-style RxRx1 JSONs as soon as they appear, then apply the
frozen 80%/50% competence gate over all four arms. Independently finish and rank the six Camelyon17
candidates. No OOD-test access, RxRx1 replication, MoE launch, or additional rescue tuning is
licensed before those gates.
