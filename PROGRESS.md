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

## 2026-07-31 19:52 EDT gate resolution and acceleration

All four bounded RxRx1 rescue arms are now strictly valid. The remaining two results are:

- `rxdiag_no_rrc_uniform_lr`: OOD-val 0.018571, seen-environment 0.052300,
  worst-environment val 0.002029, clean commit `26ad7fa`.
- `rxdiag_wilds_uniform_lr`: OOD-val 0.055003, seen-environment 0.105289,
  worst-environment val 0.010146, clean isolated commit `1da67a5`, code-equivalent to GitHub
  `aa8d0cf` for the official-style transform.

Both files pass parseability, finite-metric, exact seed/config/run identity, clean-provenance,
21,628,800-parameter, `selection_split=ood_val`, and `test_evaluated=false` checks. No OOD-test
value was constructed or consumed.

Official-style preprocessing is the clear best rescue and demonstrates that the original
normalization/augmentation mismatch was consequential. It still does not qualify the substrate.
The canonical validation-only WILDS control reached best-so-far OOD-val accuracy 0.154151 at epoch
47; the best DINOv2 rescue is 35.7% of that value. Because the predeclared rule abandons DINOv2 when
either gate metric remains below 50%, and a best-so-far validation maximum cannot decrease, the
RxRx1 DINOv2 competence gate is decisively failed without waiting for OOD test or further tuning.
Natural-image DINOv2 is frozen as excluded for RxRx1. No RxRx1 replication, router calibration,
MoE cell, or mechanism analysis will be run on it.

The recommended replacement candidate is Cell-DINO Cell Painting ViT-S/8: it is microscopy-
pretrained, close in scale to the current small ViT, and structurally compatible with FFN upcycling.
This is a recommendation, not a silently frozen design change. It requires one explicit backbone
decision and then a bounded dense competence screen before RxRx1 rejoins the factorial.

Formal Camelyon17 coverage increased to 3/6 at clean commit `26ad7fa`. The new valid candidate
`camelyon17_original_ep10_s0_hpoA_lr3e-04_llrd0.70` reached 0.800080 OOD-val and 0.987604
seen-environment accuracy. The three remaining candidates are all now active: the previously
running `1e-4/0.85` and `3e-5/0.70` cells plus `3e-4/0.85`, which was dry-run as the sole pending
shard and launched on container 2874 GPU0. The new job was independently verified at 98% GPU
utilization with the exact process identity, fresh persistent log, W&B run `fwab3bqs`, clean
`26ad7fa` checkout, and explicit OOD-test withholding.

Current critical path: finish and rank Camelyon17 Phase A while the user decides whether to qualify
Cell-DINO for RxRx1. The remaining authorized GPU work is running; idle GPUs are not filled with
unapproved backbone experiments or additional DINOv2 tuning.

## 2026-07-31 20:01 EDT artifact synchronization

The resolved RxRx1 substrate gate, complete bounded-rescue evidence, dataset-qualified-backbone
design, and refreshed research-plan HTML were pushed to GitHub through commit `23fdacd` after
safely reconciling a concurrent Overleaf file-mode-only commit. Overleaf then pulled the verified
history, reported no remaining GitHub commits, and compiled `paper/main.tex` to five pages with no
fatal error. The compiled PDF contains the dataset-qualified abstract and the 5.50% RxRx1 rescue
result. GitHub/local remains the source of truth.

## 2026-07-31 21:31 EDT Cell-DINO qualification implementation

The user approved Meta Cell-DINO as the RxRx1 replacement substrate. The project is now RxRx1-only
for new scientific work; no new Camelyon17 jobs are licensed, although existing healthy jobs and
their artifacts are preserved.

The RxRx1 kill-test protocol, Cell-DINO adapter, fixed three-to-five-channel mapping, competence
launcher, tests, and exact matched-control accounting were pushed to GitHub main at `db9ebdb`.
Because SciServer cannot authenticate to the private GitHub repository, the exact patch was applied
to an isolated execution checkout at
`/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation-run-db9ebdb-v2`. Its tree hash
is `c537d33ef4db3dc2206c0922a0159b4b19095adc`, exactly matching GitHub `db9ebdb`; the resulting
isolated SciServer commit is `4c1a0ab2b8005c6f1e7910ff5fa90b2b6b26e856`.

The official Meta DINOv2 source was cloned to persistent storage and pinned at
`7764ea0f912e53c92e82eb78a2a1631e92725fc8`. The complete remote test suite passes: 80/80 tests in
7.58 seconds, including Cell-DINO chunked-block loading, five-channel input, fixed RxRx1 channel
mapping, classifier-inclusive parameter accounting, and existing leakage/protocol guards.

No Cell-DINO GPU run has been launched yet. The approved checkpoint is not present at the frozen
persistent destination
`models/cell_dino/cell_dino_vits8_pretrain_cp-37d20e9c.pth`. The currently inspected SciServer
container has two idle H100s and no MoE/Cell-DINO process. The browser-provided signed link could
not be downloaded automatically without bypassing browser security, so the exact checkpoint file
must be downloaded or uploaded by the user once. This is the sole blocker to the three-run
competence diagnostic (frozen linear probe; full fine-tuning at `1e-4`; full fine-tuning at
`3e-4`). OOD test remains untouched.

The 15-minute research-steward automation was updated to the RxRx1 Cell-DINO diagnosis, competence,
kill-contrast, replication, mechanism, exact-parameter-fairness, and traceable-reporting protocol.
Once the checkpoint appears, it is authorized to integrity-check it, dry-run all three diagnostics,
fill idle GPUs without duplicates, validate train/ID/OOD-validation outputs, and advance to the
matched dense-wide versus MoE kill contrast only if dense competence is established.
