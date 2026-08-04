# Progress ledger

Last verified: 2026-08-04 06:42 EDT on SciServer and locally. Ten of ten currently available H100s
are assigned to five matched learned/frozen epoch-5 pairs; tester6 remains scheduler-Pending and
contributes no GPU. Twelve additional short arms are exact-built and runnable. The paper has not
been changed because the new results remain exploratory seed-0 mechanism evidence.

Research-state synchronization: GitHub commit `75c2e85`, containing the validated negative kill
gate and manuscript table, was pulled into the linked Overleaf project on 2026-08-01;
`paper/main.tex` compiled successfully to five pages with 0 errors and one pre-existing warning.
Overleaf then reported no newer GitHub commit since the merge.

## 2026-08-04 06:42 EDT — corrected router pairs validate; five new endpoint pairs refill the pool

- All ten corrected/new epoch-5 rows are strict-valid: finite metrics, exact clean execution commit
  `818fc8a`, blocks 10+11, four environments/9,854 OOD-validation samples, complete checkpoints,
  `selection_split=ood_val`, `test_evaluated=false`, and null OOD-test fields. The six earlier
  configuration-mismatch attempts remain preserved exclusions and are not pooled.
- The largest mean learned-minus-frozen result is E8 linear top-1 at `+0.599` OOD-validation and
  `+0.926` ID points, but its worst environment falls `0.081` points and randomized routes perform
  better (`-0.00132` reliance gap). E16 cosine top-2 is the only aligned pair on mean, tail, and
  route reliance, but the changes are only `+0.101/+0.244` OOD/worst points and `+0.00142`
  reliance. No pair reaches the predeclared `0.01` causal-routing signal, so none continues.
- Every released GPU was refilled with an exact matched endpoint pair: E16 linear top-2 on 2887,
  E32 linear top-1 on 2875, E2 linear top-1 on 2874, E4 noise `0.001` on 2862, and E16 noise
  `0.001` on 2859. Each container uses GPU0 for learned and GPU1 for frozen; ten are active and
  zero are idle.
- The next queue is again 12 runnable arms after ten real Cell-DINO model builds passed: E2/E32
  linear top-2, E2/E32 noise `0.001`, and E4 image routing, each learned/frozen. The first audit
  queried `top_k` from the wrong report object; a metadata-only repair reads it from the loaded
  config, after which all 10/10 builds passed with unique run IDs and no data or OOD-test access.
- Fixed expert partitioning, ordinary optimization, or short seed-0 trajectory noise remains more
  plausible than useful adaptive routing. The sharp falsifier is an endpoint with route reliance
  above `0.01` plus aligned learned-over-frozen OOD mean and worst-environment improvement.
  W&B is live, HF retry remains quota-blocked, and the paper is unchanged. Trace:
  `analysis/fast_conflict_router_dynamics_epoch5_wave7_corrected_linear_top2_validation.json` and
  `analysis/fast_conflict_ready_queue_hb8_extension_validation.json`.

## 2026-08-03 20:21 EDT — fast architecture tooling passes the full suite; 30 becomes the search ceiling

- The single-FFN limitation is removed prospectively. A run can now replace explicit lists of
  transformer FFNs, while its dense comparator modifies the same indices and remains within the
  predeclared 0.1% total-parameter tolerance at study scale. Run identity, capacity records,
  environment handling, auxiliary loss, routing audits, and randomized-route controls all cover
  every converted block without breaking the historical single-block interface.
- A training-only profiler now ranks all 12 Cell-DINO FFNs by cross-experiment gradient cosine,
  conflict rate, gradient norm, and balanced-minibatch uncertainty. It records a high-conflict
  target and low-conflict placebo without reading OOD validation or OOD test.
- On the clean SciServer test checkout, 37 focused tests and all 106 repository tests pass. The
  profiler dry-run is parseable and asserts `data_scope=train_only`, `test_evaluated=false`, and
  null OOD-test outcomes. This is validated engineering, not a new accuracy result. Trace:
  `analysis/fast_discovery_engineering_validation.json`.
- The prospective schedule is now 2/5/10/20/30. Ten epochs is the cheapest credible filter but not
  a safe negative answer because the canonical MoE-minus-dense trajectory moved from `-0.416`
  points at epoch 10 to `+2.781` at epoch 30. Epoch 20 is the new interpolation probe; epoch 30 is
  the maximum exploratory horizon. Linearized one-H100 estimates are 26--32 minutes for 10,
  53--65 minutes for 20, and 79--97 minutes for 30.

## 2026-08-03 19:55 EDT — prospective search pivots to fast mechanistic discovery

- The user explicitly deprioritized additional broad 60--90 epoch exploration. Completed evidence
  is unchanged, but future allocation now uses saved-checkpoint diagnostics and 2/5/10-epoch
  screens. A healthy job already near a declared checkpoint may finish; a stalled long exploratory
  arm is not restarted merely for matrix completeness.
- The first question is where gradients from different training experiments conflict across all 12
  Cell-DINO FFNs. The next bounded architecture comparison is one versus two sparse FFNs placed at
  measured conflict peaks, each paired with learned-router, frozen-router, and equal-budget dense
  controls. Staged router/expert unfreezing is crossed only after the exact smoke tests pass.
- Accuracy alone does not decide at epoch 2 or 5 because the canonical sparse trajectory emerged
  late. An arm must show causal routing: route reliance at least 1 point, multiple materially used
  experts, reproducible conflict reduction, and either paired mean/tail movement or a predeclared
  tail signature. Only then may it reach epoch 30; epoch 60/90 is no longer default search compute.
- This is a protocol change, not a new result or live pool audit. OOD validation remains the
  exploratory selection split, OOD test remains sealed, and exact-total-parameter versus
  active-compute claims remain separate. Trace:
  `analysis/fast_mechanistic_discovery_protocol.json`.

## 2026-08-03 16:56 EDT — upcycling wave two, first frozen controls, and six-arm refill

- Six new epoch-60 rows pass exact 10/30/60 milestone, finite-metric, four-environment/9,854-sample,
  checkpoint, controller-exit, fatal-scan, `selection_split=ood_val`, and `test_evaluated=false`
  checks: the final three upcycling arms, frozen E4 and E8 controls, and canonical E4 no-auxiliary.
  All eight predeclared noisy-upcycling arms are now strict-valid.
- The route E16 noisy arm minus its exact zero-noise anchor is only
  `+0.142/-0.128/+0.284` OOD-validation/ID/worst-experiment points. Learned E4 noise `0.01`
  minus frozen routing is `+0.375/+0.785/-0.122`; learned E8 noise `0.001` minus frozen routing is
  `+1.005/+0.628/-0.244`. The learned routers retain small mean advantages but lose tail accuracy,
  while prior route reliance is below `0.01`. This does not support useful conditional
  specialization and licenses neither epoch 90 nor fresh seeds.
- Canonical E4 tail-safe minus no-auxiliary is `+0.203/+0.254/+0.203` points, a small aligned
  objective effect well below a material result. Exact zero-noise comparators for canonical E8 and
  E16 remain active, so those noisy-arm effects are not claimed.
- Six released GPUs were dry-run as exactly one unique pending cell each and refilled with dense E8
  noise `0.001`, frozen/dense E8 noise `0.01`, and learned/frozen/dense E16 noise `0.001`
  controls. Together with the repaired dense E4 retry and three zero-noise anchors, all ten
  available H100s are active; zero are idle. Ready depth remains 12 and backlog depth 26.
- tester6 replacement 2899 is still authoritative-detail `Pending` with the unchanged
  affinity/unschedulable/insufficient-GPU reason; the dashboard's stopped label did not trigger a
  duplicate start. OOD test remains sealed, W&B is live, HF retry remains quota-blocked, and the
  paper is unchanged. Trace:
  `analysis/upcycling_noise60_epoch60_wave2_and_controls_validation.json`.

## 2026-08-03 07:43 EDT — moderate-bank epoch 30 and canonical E64 milestone

- All eight moderate-bank epoch-10 rows and seven currently available epoch-30 rows pass strict
  parseability, finite-metric, exact run/epoch, checkpoint, fatal-scan, `selection_split=ood_val`,
  and `test_evaluated=false` checks. The only missing epoch-30 row is canonical E4 no-auxiliary,
  which is still training and is therefore not excluded or adjudicated.
- At epoch 30, tail-safe minus exact no-auxiliary for route E4 is
  `+0.436/-0.007/+0.731`, route E8 is `+1.969/+1.433/-0.406`, and route E16 is
  `+1.350/+0.987/-0.244` OOD-validation/ID/worst-experiment points. Each pair retains a distinct
  mean-or-tail signal, so all continue to the predeclared epoch-60 mechanism adjudication. None is
  near the `+5` target; no epoch-90 or fresh-seed promotion is licensed.
- The canonical E64 temperature-0.3 pair also validates at epochs 10 and 30. At epoch 30,
  tail-safe minus no-auxiliary is `+0.041/-1.396/-0.162` points: effectively tied on mean OOD with
  ID and worst-experiment losses. The locked pair continues to epoch 60 for its predeclared
  trajectory/mechanism check, without promotion.
- Ten of ten available H100s remain assigned across containers 2887, 2875, 2874, 2862, and 2859;
  zero are idle. The ready queue remains 12 runnable arms and the hypothesis backlog remains 26.
  tester6 replacement 2899 is still scheduler-Pending for the same affinity/unschedulable/GPU
  constraints and contributes no available device.
- OOD test remains sealed, W&B streams are live, HF publication was not retried under the known
  private-storage quota blocker, and the paper is unchanged for this multiplicity-exposed seed-0
  milestone. Trace:
  `analysis/temperature03_moderate_epoch30_and_canonical_E64_validation.json`.

## 2026-08-03 06:48 EDT — canonical E32 final, moderate-bank epoch 10, and refill

- Strict validation closed the canonical E32 temperature-0.3 tail-safe/no-auxiliary pair at epoch
  60. Tail-safe minus its exact active-compute comparator is `+0.457` OOD-validation points,
  `+0.488` ID points, and `+0.284` worst-experiment points. One of four environments worsens
  (`-0.731` points), the epoch-30 tail contrast was negative, and route reliance is only `0.00132`
  versus `0.00294`. This is a small aligned exploratory difference, not a material, consistent, or
  routing-mediated effect; it licenses neither epoch 90 nor fresh seeds.
- Seven available moderate-bank epoch-10 rows and checkpoints pass strict parseability, finite
  metric, run/epoch identity, fatal-scan, `selection_split=ood_val`, and
  `test_evaluated=false` checks. Route tail-safe minus no-auxiliary OOD/ID/worst differences are
  E4 `+0.345/-0.498/-0.122`, E8 `-0.731/-1.770/-0.244`, and E16
  `+0.599/-0.399/+0.122` points. No row is promoted. The predeclared delayed-emergence exception
  permits these cells only through epoch 30, where unchanged dominance or a missing competitive
  mechanism trajectory causes pruning.
- Container 2859 GPU 0 was immediately refilled with the exact canonical E4 temperature-0.3
  no-auxiliary comparator (controller `60873`, worker `60878`) after exact shard `7/12` dry-run,
  duplicate/result/marker, persistence, tracking, checkpoint, and sealed-test guards passed. It is
  paired with canonical E4 tail-safe on GPU 1. All ten available H100s are assigned; zero are idle.
  The runnable queue is 12 arms (four remaining moderate-bank cells plus eight tested upcycling
  cells), with 26 hypotheses in backlog. tester6 replacement 2899 remains scheduler-Pending and
  contributes no available GPUs.
- The OOD test remains sealed. W&B streams are live. HF publication was not retried because the
  known private-storage quota blocker has not cleared. The manuscript was not changed for this
  multiplicity-exposed seed-0 terminal negative and diagnostic milestone.
- Trace: `analysis/temperature03_canonical_E32_epoch60_and_moderate_epoch10_refill_validation.json`.

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

## 2026-07-31 22:48 EDT Cell-DINO checkpoint recovery and competence launch

The approved Cell-DINO CP ViT-S/8 upload arrived in two consecutive SciServer pieces: a
48,234,496-byte prefix with the PyTorch ZIP header and a 37,929,888-byte suffix. Neither fragment
loaded alone. They were concatenated non-destructively into the frozen checkpoint destination,
producing an 86,164,384-byte file with SHA-256
`37d20e9cd48b3d610b5de15a4ea4e7e060a593b8d8358e928d079dc7b03ee66a`. The digest begins with the
`37d20e9c` identifier in Meta's approved filename. `torch.load(..., weights_only=True)` succeeds and
exposes the expected DINOv2 state-dict keys.

An end-to-end model smoke test then loaded the checkpoint through the pinned official repository,
flattened 12 actual transformer blocks, accepted a `1x5x128x128` tensor, and produced logits of
shape `1x1139`. The original model has 21,964,787 total parameters, and its recorded checkpoint
hash matches the verified digest. The complete remote suite had already passed 80/80 tests.

The competence launcher dry-run reported exactly three pending diagnostics and no existing result:
the five-epoch frozen linear probe, ten-epoch full fine-tuning at `1e-4`, and ten-epoch full
fine-tuning at `3e-4`. Both GPUs in the inspected container were idle and no duplicate Cell-DINO
worker existed. The launcher started the first two jobs on GPUs 0 and 1 and retained the `3e-4`
job in its idempotent queue. Independent health verification found the two direct workers alive,
both H100s at 99% utilization (about 2.0 and 7.9 GiB respectively), fresh persistent logs, and live
W&B runs `lvyxc7my` (linear probe) and `y82kwuxm` (full fine-tuning `1e-4`). Dataset initialization
records 33 training experiments and only train, ID-test, and OOD-validation sizes; the OOD test is
explicitly untouched.

These runs are diagnostic rather than performance evidence. The next gate is to validate all three
JSONs and classify the failure regime from train, seen-environment, OOD-validation, and
worst-experiment accuracy. The matched dense-wide versus MoE kill contrast remains unlicensed until
Cell-DINO demonstrates credible dense competence.

## 2026-07-31 23:02 EDT First Cell-DINO diagnostic validated

The frozen-backbone linear probe completed and passed strict validation at the isolated SciServer
commit `4c1a0ab2` (result provenance records the unique short form `4c1a0ab`). The file is parseable,
all required metrics are finite, the run/config/seed and Cell-DINO checkpoint identities match, the
pinned DINOv2 source is `7764ea0`, `selection_split=ood_val`, and `test_evaluated=false`.

Its train, seen-environment, OOD-validation, and worst-experiment accuracies are 5.83%, 4.05%,
2.87%, and 0.97%. This is diagnostic evidence that a frozen linear readout is insufficient; it is
not evidence against full fine-tuning or sparse conditional capacity. The launcher immediately
filled the freed slot with the `3e-4` full-fine-tuning arm. At the fresh health check, the `1e-4`
and `3e-4` jobs were at epochs 9 and 4 with fresh losses 5.0466 and 6.6335. Both assigned H100s were
at 99% utilization and approximately 7.9 GiB; all six H100s in the other three containers were
confirmed idle. No additional competence run is scientifically licensed until these two complete.

The next gate is to validate both full-fine-tuning JSONs and decide whether Cell-DINO establishes a
credible seen-environment result with a measurable OOD-validation gap. Only then is the seed-0
original versus exact-total-parameter-matched dense-wide versus canonical-MoE kill contrast
licensed.

## 2026-07-31 23:05 EDT Cell-DINO full fine-tuning `1e-4` validated

The `1e-4` full-fine-tuning result completed and passed the same strict schema, finite-metric,
run/config/seed, checkpoint/source-provenance, OOD-validation selection, and test-blindness checks.
Its train, seen-environment, OOD-validation, and worst-experiment accuracies are 14.46%, 8.59%,
4.84%, and 1.38%. Full adaptation therefore improves materially over the frozen linear probe, but
the absolute level is still weak. This is provisional until the predeclared `3e-4` arm finishes;
the competence gate must use both full-fine-tuning results rather than stopping on the first one.

The `3e-4` job remains healthy at epoch 5 with a fresh loss of 6.4795. It is the only active GPU
worker, so seven of the eight H100s are currently idle. No extra diagnostic or MoE run is licensed
until this final competence result resolves whether Cell-DINO is a credible dense substrate.

## 2026-07-31 23:15 EDT Frozen out-of-the-box representation probe launched

The user authorized one additional bounded diagnostic to determine whether Cell-DINO already
organizes RxRx1 morphology usefully before any supervised adaptation. Because Cell-DINO is
self-supervised and has no 1,139-way RxRx1 head, a literal zero-shot class prediction is undefined.
The implemented test freezes every model weight and evaluates two deterministic readouts over the
same embeddings: exact cosine 1-nearest-neighbour and nearest perturbation centroid. The former
tests local feature retrieval; the latter averages each class across training experiments and is
less able to exploit batch-local neighbours.

This comparison has a direct batch-effect interpretation. High ID-test but low OOD-validation
1-NN accuracy would indicate that the embedding space is useful within familiar experiments but
organized around acquisition context. Low ID and OOD for both readouts would indicate that the
pretrained representation does not expose the perturbation classes without adaptation. Better
centroid than 1-NN OOD accuracy would show that averaging across batches suppresses nuisance
structure even before model training.

The official Cell-DINO paper describes the CP checkpoint as self-supervised on a combined resource
drawn from five Cell Painting datasets and explicitly names RxRx-series models as future work. We
therefore found no documented direct RxRx1 pretraining exposure. The domain is intentionally
similar, but the benchmark labels and RxRx1 images are not documented as pretraining data.

The new probe and two focused regression tests were transferred to an isolated checkout after four
focused tests and the complete 82-test suite passed. The script SHA-256 is
`2541cdff100e139222225d5c617965eac54e81bba7ba895ceb7e641287ecd33b`. It launched on container 2874
GPU0 after a no-result, no-duplicate, two-idle-GPU preflight. Independent health verification found
the exact probe process, 70% utilization and 7.1 GiB on GPU0, a fresh persistent log, 33 training
experiments, and only train/ID-test/OOD-validation sizes. OOD test remains untouched. The output is
`kill_rxrx1/oob/rxrx1_cell_dino_frozen_oob_readouts_s0.json`.

## 2026-07-31 23:26 EDT Cell-DINO out-of-box and competence diagnostics complete

The first frozen probe embedded 40,576 of 40,612 training examples because the optimization loader
intentionally drops its final partial batch. That output was preserved as
`rxrx1_cell_dino_frozen_oob_readouts_s0.partial-40576.json` and excluded. The probe now rebuilds a
deterministic evaluation loader with `drop_last=false`; a focused regression test covers the exact
failure. The corrected script is GitHub commit `9f56c99`, has SHA-256
`26fe7b227823d174efa85ce2c61ff4518baf95303be69c994955b2bd5058dcc6`, embeds all 40,612 training
images, and passes 3/3 focused plus 83/83 full-suite tests in the isolated SciServer checkout.

The corrected, strictly valid non-parametric result is:

| Frozen readout | ID-test accuracy | OOD-validation accuracy | worst OOD-validation experiment |
|---|---:|---:|---:|
| cosine 1-nearest-neighbour | 0.025214 | 0.012279 | 0.004870 |
| nearest class centroid | 0.017138 | 0.007611 | 0.001623 |

The final full-fine-tuning diagnostic (`lr=3e-4`) also completed and passed every run/config,
checkpoint, DINOv2-source, finite-metric, split, and test-blindness check. Its train/ID/OOD-val/
worst-experiment accuracies are `0.089314 / 0.060672 / 0.037447 / 0.011769`, weaker than the valid
`lr=1e-4` arm (`0.144568 / 0.085935 / 0.048407 / 0.013799`). All three predeclared competence arms
are now valid. OOD test remains untouched.

This resolves the failure-regime gate as representation/optimization failure under the current
recipe, not demonstrated batch-transfer failure: training accuracy itself remains only 14.5% in the
best adapted model, and the frozen embeddings provide only 2.5% ID 1-NN accuracy. The OOD decline is
real but secondary; the model is not yet competent enough within seen experiments to test whether
MoE improves unseen-batch transfer. No MoE kill contrast is licensed. The next bounded work is to
audit Cell-DINO's exact channel order, normalization, official evaluation recipe, and optimization
budget against its released implementation before deciding whether to retain the checkpoint.

## 2026-07-31 23:44 EDT Official Cell-DINO representation repaired and rerun

The implementation audit found a genuine representation mismatch, not a new tunable
hyperparameter. The Cell-DINO paper specifies a downstream representation formed by concatenating
the normalized class token with the mean of the normalized last-block patch tokens. Our initial
wrapper classified from the class token alone. The earlier results remain valid for that CLS-only
instrument, but they are superseded for Cell-DINO qualification.

GitHub commit `04986fe` implements and records `cls_patch_mean`. The isolated SciServer checkout is
the code-equivalent commit `f57743d` with tree
`289390c317910c0a1f8839832405c80f32b5ac3e`. All 84 tests pass there. A real-checkpoint smoke test
maps a `2 x 5 x 128 x 128` input to a `2 x 768` representation and `2 x 1139` logits, with a
768-input classifier and checkpoint/source/pooling provenance recorded.

The same three competence arms were relaunched without overwriting the earlier outputs, under
`kill_rxrx1/official_pool/competence/`. The frozen linear probe is healthy on container 2874 GPU 0
and has reached epoch 2; the `1e-4` and `3e-4` full-fine-tuning arms are healthy on the second
container's GPUs 0 and 1. Three H100s are active and five are idle because no additional pre-gate
arm is licensed. OOD test remains untouched.

The instrument decision is reopened only for this exact correction. The old best still fails. If
the official pooled representation remains weak, the main remaining limitation is the three-to-five
channel mapping, which permanently zeros two pretrained stain slots; that would motivate changing
the data representation or substrate rather than opening an unconstrained recipe sweep.

## 2026-07-31 23:47 EDT Official-pooling linear probe validated

The corrected frozen-backbone linear probe completed and passed strict parseability, finite-metric,
run/config/seed, clean-code, total-parameter, checkpoint/source, feature-pooling, split, and
test-blindness checks. Its train/seen/OOD-validation/worst-experiment accuracies are
`0.075168 / 0.046070 / 0.030343 / 0.008523`. Relative to the superseded CLS-only linear probe,
official pooling changes these by `+0.016882 / +0.005590 / +0.001624 / -0.001218` absolute.

This is a validated diagnostic, not a competence decision. Concatenating the mean patch token adds
some linearly accessible training and seen-environment signal, but almost no OOD-validation gain;
the frozen representation remains far below a useful instrument. The two full-fine-tuning arms are
still the decisive test of whether joint adaptation can exploit the corrected representation.

At the fresh health check, full fine-tuning at `1e-4` had reached epoch 4 and `3e-4` epoch 5, both
with advancing persistent logs and both assigned GPUs at 100% utilization with about 7.9 GiB each.
No fatal error was present. Two H100s are active and six authorized H100s are idle because no
additional pre-gate arm is scientifically licensed. OOD test remains untouched.

## 2026-08-01 00:31 EDT native-channel instrument pivot

The corrected official-pooling competence set is complete and strictly valid. The best arm is
full fine-tuning at `1e-4`, with train/seen/OOD-validation/worst-experiment accuracies of
`0.177642 / 0.111642 / 0.059265 / 0.012581`. The `3e-4` arm reaches
`0.065507 / 0.050650 / 0.031967 / 0.009740`; the frozen probe reaches
`0.075168 / 0.046070 / 0.030343 / 0.008523`. Every result has clean SciServer commit `f57743d`,
the expected checkpoint and DINOv2 source, official `cls_patch_mean` pooling,
`selection_split=ood_val`, and `test_evaluated=false`.

The best corrected instrument reaches only 38.4% of the canonical OOD-validation reference and
45.4% of its seen-environment reference. It therefore fails the frozen competence boundary. This
is decision-grade evidence against the *three-channel WILDS composite to zero-filled CP5
interface*, not against Cell-DINO on native stains and not against MoE. The MoE kill comparison
remains unlicensed.

The user authorized the minimal native-channel diagnosis. The official 49,039,640,485-byte RxRx1
archive downloaded to persistent storage and is being extracted by four disjoint cell-type
workers. No GPU is occupied while data preparation runs. The frozen two-instrument design is:

| Instrument | Pixel interface | Seed-0 diagnostic pair | Status |
|---|---|---|---|
| Cell-DINO CP ViT-S/8 | `[w1,w2,w4,mean(w3,w6),w5]` = `[DNA,ER,RNA,AGP,Mito]` | frozen probe + full FT `1e-4` | queued after complete train/ID/val audit |
| Channel-Adaptive DINO ViT-L/16 | native `w1..w6` Bag-of-Channels | frozen probe + full FT `1e-4` | code ready; approved checkpoint file absent |

GitHub commits `505f360` and `c01d7c6` implement the native loader, joint-channel geometry,
biological mapping, Channel-Adaptive DINO adapter/config, and parameterized idempotent launcher.
The isolated SciServer execution commit is `8624481`, tree
`343003803975f40260975ea32c763c389c11a8da`. Thirteen focused tests and the full 88-test suite pass.
OOD test was neither constructed nor evaluated. The next automatic action is to finish extraction,
run the strict selection-split coverage/image audit, smoke the real checkpoint on native tensors,
dry-run the two Cell-DINO shards, and launch them on both idle H100s. The Channel-Adaptive pair can
launch immediately after its separately approved checkpoint appears at the frozen persistent path.

## 2026-08-01 00:56 EDT native data and checkpoint preflight

The first selection-split audit correctly refused to pass because the initial parallel extraction
had produced only 293,748 of the archive's 753,319 files. This is an orchestration defect, not a
scientific or mapping result: the supposedly missing experiment and channel paths were verified to
exist inside the intact 49,039,640,485-byte archive, and persistent storage has ample free space.
A single skip-existing extraction is now running as PID `34613`; it preserves completed files and
is filling the remaining archive deterministically. The partial audit is excluded from all model
decisions and will be rerun from scratch before launch. OOD test remains unevaluated.

The native Cell-DINO interface passed an independent real-checkpoint smoke on the isolated clean
execution tree: a six-channel `6 x 512 x 512` raw sample was mapped to the frozen
`[w1,w2,w4,mean(w3,w6),w5]` CP5 tensor at `5 x 128 x 128`, loaded with the released checkpoint and
official `cls_patch_mean` feature, and produced finite `1 x 1139` logits on an idle H100. The
latest isolated execution identity is SciServer commit `510bccf`, tree
`88e356a061d8dc1dbaa61a98bb889d90518d3b51`; 89/89 tests pass and dry-run manifests have distinct
Cell-DINO and Channel-Adaptive run namespaces. The Cell-DINO linear/full-FT pair remains queued
until the complete train/ID-test/OOD-validation channel audit passes. The native-six
Channel-Adaptive pair remains blocked only on its distinct approved ViT-L/16 checkpoint; the
existing CP ViT-S/8 checkpoint cannot substitute for it.

The manuscript was refocused from the former broad factorial narrative to the RxRx1 competence,
matched-capacity kill contrast, and mechanism sequence. It compiles with fatal errors enabled and
was pushed to GitHub as commit `48f28dd`.

## 2026-08-01 01:12 EDT Channel-Adaptive comparison preflight

The user confirmed that Channel-Adaptive DINO should remain in the bounded native-channel
instrument comparison. A correctness audit found that the first adapter implementation did not use
Meta's official Bag-of-Channels feature route: each acquisition must be encoded independently by
the shared one-channel ViT and the per-channel CLS (plus optional mean patch) features concatenated.
GitHub commit `d46f684` repairs that path and adds a regression test. The fix was transferred to the
isolated SciServer execution tree as clean commit `7365eac2262735e339c74f506296e07f4e47512ce`,
tree `4f551c7f701c69392f8ff4950b34b523a31de54c`; 15 focused tests and the full 90-test suite pass.
No Channel-Adaptive metric exists yet, so this changes implementation trust rather than scientific
interpretation.

The resumed native archive extraction remains healthy as PID `34613`: 544,697 files were present
after 23:45 elapsed, up from 484,778 at the previous fresh signature, with no fatal error. The
archive contains 753,319 files, so the complete train/ID-test/OOD-validation audit is still pending
and no native run is licensed yet. All four running SciServer containers were inspected directly:
all eight H100s are idle and no MoE/RxRx1 worker is present. Filling GPUs with duplicate seeds before
the instrument gate would not shorten the decision path. When extraction and
the strict audit complete, the Cell-DINO frozen/full-FT pair will launch on the two attached H100s.
The Channel-Adaptive frozen/full-FT pair is additionally blocked on its distinct approved ViT-L/16
checkpoint at the frozen persistent path; the CP ViT-S/8 checkpoint is not interchangeable.

## 2026-08-01 01:38 EDT zero-idle-handoff launch watcher armed

Native extraction remains healthy and advanced to 639,036/753,319 files at 47:50 elapsed; PID
`34613` is alive in I/O wait, the count continues to increase, and no fatal error is present. Both
GPUs in the execution container are idle and the distinct Channel-Adaptive checkpoint is still
absent.

To avoid losing up to one heartbeat interval after the data becomes complete, a one-shot persistent
watcher was armed as PID `35763` with log `kill_rxrx1/native_cp5_watcher.log`. It waits for the exact
extraction PID to exit, then requires a clean tested execution tree, the released Cell-DINO
checkpoint, a zero-missing train/ID-test/OOD-validation audit, a two-run dry-run manifest, no
duplicate launcher, and zero GPU compute processes. Only if every gate passes does it launch the
native CP5 frozen-probe/full-fine-tuning pair on GPUs 0 and 1. The watcher was independently
verified alive; it has not launched training and has not accessed OOD test. A failed audit or
occupied GPU stops the chain rather than weakening a gate.

## 2026-08-01 02:33 EDT native CP5 competence pair launched

Classification is `RUNNING_HEALTHY`. The resumed extraction completed, and the fresh strict native
audit passed over exactly 91,078 selection samples: 40,612 train, 40,612 ID-test, and 9,854
OOD-validation samples, each with all six channels present. The 66-file pixel smoke found `L`-mode
`512 x 512` images. The manifest records `test_evaluated=false`; OOD test was not loaded or scored.

The one-shot watcher successfully ran the audit and two-run dry-run, then stopped at its duplicate
guard because the guard matched the watcher's own command line. This was a narrow handoff defect,
not a data, model, or scientific failure. Before launching directly, the steward re-verified the
clean isolated SciServer commit `7365eac2262735e339c74f506296e07f4e47512ce`, zero active
Python training processes, two idle GPUs, the frozen two-run manifest, checkpoint presence, and the
zero-missing audit. No gate was weakened and no duplicate was launched.

Both frozen seed-0 Cell-DINO native instruments are now healthy:

| Run | GPU / PID | Fresh progress | W&B |
|---|---|---|---|
| `cell_dino_native_cp5_instrument_linear_probe` | GPU 0 / `36861` | epoch 0 recorded; loss `7.3955`; 1.97 GiB, 70% utilization at the fresh signature | `1sfzdw4j` |
| `cell_dino_native_cp5_instrument_full_ft_lr1e-4` | GPU 1 / `36862` | epoch 0 recorded; loss `7.2293`; 7.87 GiB, 99% utilization at the fresh signature | `7rdhcisv` |

The initial losses prove only that both training loops are advancing; they are not performance
evidence. Zero of two native result JSONs exists yet. Two of eight authorized H100s are occupied;
the other six remain idle because the Channel-Adaptive pair still lacks its distinct approved
ViT-L/16 checkpoint, and neither extra seeds nor an MoE comparison is licensed before this
competence result. The next automatic action is strict validation of each completed JSON and the
train/ID/OOD-validation failure-regime classification. A competent native dense instrument licenses
the frozen original versus exact-total-parameter-matched dense-wide versus canonical-MoE kill
contrast; a floor result stops that launch and diagnoses the instrument instead.

## 2026-08-01 03:14 EDT native competence passes and kill contrast launches

Classification is `RUNNING_HEALTHY`. Both native CP5 competence JSONs completed and passed strict
parseability, finite-metric, filename/run/config/seed, clean-tree, checkpoint/source, parameter,
split, and test-blindness checks. The frozen linear probe reaches train/ID/OOD-validation/worst-
experiment accuracies of `0.106245 / 0.076406 / 0.044246 / 0.008117`. Full fine-tuning at
`1e-4` reaches `0.390009 / 0.270930 / 0.122894 / 0.014205`. Relative to the frozen probe, full
adaptation gains 28.38 train points, 19.45 ID points, and 7.86 OOD-validation points.

The full-fine-tuned model reaches 79.72% of the canonical WILDS control's best validation accuracy
(`0.122894 / 0.154151`), 91.64% of its epoch-21 validation accuracy, and 110.22% of its recorded
epoch-21 ID accuracy. This passes the frozen dense-competence rule: the model is learnable and in
the canonical sanity range. The failure regime is therefore genuine batch transfer rather than a
floor substrate: train is 39.00%, ID is 27.09%, and OOD validation is 12.29%, a 14.80-point
ID--OOD gap. The 1.42% worst-experiment score additionally shows severe held-out-experiment
heterogeneity. These are decision-grade instrument-qualification findings, not evidence that MoE
helps.

The real Cell-DINO parameter audit instantiates 22,402,163 parameters for the original model,
30,675,834 for dense-wide, and 30,676,212 for MoE. Dense-wide and MoE differ by only 0.001232%,
within the frozen 0.1% tolerance; their active FFN parameters are 9,455,239 and 1,184,641,
respectively. The three seed-0 kill shards dry-ran as the only pending runs with the same native
data, checkpoint, seed, optimizer, schedule, ten-epoch budget, and `1e-4` learning rate.

Three H100 workers are active. Container 2875 GPU 0 runs
`rxrx1_dense_wide_middle_canonical_E8_ep10_s0_kill_dense_wide` (W&B `g9ucrbln`), and GPU 1 runs
`rxrx1_moe_middle_token_cosine_canonical_E8k1_ep10_s0_kill_moe` (W&B `u6s1sms3`); both have fresh
epoch-1 logs, high utilization, and no fatal error. The first original-arm launch attempted a
second Jupyter workspace that mapped to the same physical container; the global GPU lease correctly
rejected it before model or W&B start. This excluded orchestration failure was retried once on
freshly inspected container 2874 GPU 0. The retry, `rxrx1_original_ep10_s0_kill_original` (W&B
`8l3xts1p`), has a live process, GPU allocation, and fresh W&B start with no result yet. The other
five H100s are idle because extra seeds are forbidden until the seed-0 gate and the distinct
Channel-Adaptive checkpoint is still absent.

OOD test remains untouched. The next automatic action is to strictly validate all three kill JSONs
and compute the predeclared conditional gain, MoE OOD-validation minus dense-wide OOD-validation.
Seeds 1 and 2 launch only if that gain is at least five absolute points and MoE loses no more than
two ID points; otherwise the architecture grid stops and the negative contrast is analyzed.

## 2026-08-01 04:04 EDT seed-0 kill gate fails; bounded route diagnosis launched

Classification is `RUNNING_HEALTHY`. All three seed-0 kill JSONs completed and passed strict
parseability, finite-metric, filename/run/config/seed, code/tree, checkpoint/source, parameter,
per-environment, split, and test-blindness validation. The exact-total-parameter-matched result is:

| Model | Train | ID | OOD-val | Worst experiment | Total parameters | Active FFN parameters |
|---|---:|---:|---:|---:|---:|---:|
| Original Cell-DINO | 0.391216 | 0.269896 | 0.123097 | 0.015016 | 22,402,163 | 1,181,568 |
| Dense-wide | 0.361347 | 0.243647 | 0.104628 | 0.012175 | 30,675,834 | 9,455,239 |
| Canonical MoE | 0.387224 | 0.263469 | 0.116095 | 0.013799 | 30,676,212 | 1,184,641 |

MoE improves OOD validation over dense-wide by `0.011467` (1.15 points), ID by `0.019822`, and
worst-experiment accuracy by `0.001623`. It improves all four held-out experiments relative to
dense-wide, but the smaller original model remains better than MoE by 0.70 OOD-validation points.
Dense-wide and MoE differ by only 0.001232% in total parameters. The predeclared replication trigger
requires at least +5.0 OOD-validation points with no more than a 2.0-point ID loss; the observed
gain therefore **fails the kill gate**. Seeds 1 and 2, router calibration, and an architecture grid
are not licensed.

This is a decision-grade negative gate at seed 0, not a claim that the true effect is exactly zero.
It rejects the targeted 10--15-point conditional-capacity effect for this canonical configuration
strongly enough to stop expansion under the frozen rule. The all-four-experiment direction is a
useful diagnostic, but its small magnitude and single seed do not establish a general robustness
benefit. Hypothesis H3 (robustness beyond capacity) is not supported: MoE does not beat the smaller
original model, and adding dense width is actively harmful here.

The only licensed post-gate computation is the bounded frozen/random-route diagnosis. A clean
isolated SciServer execution commit `1344e4d` (tree
`b51a84000aa67cb97f142b084b9b90b7d25b5a2d`) launched two seed-0 Stage-2 diagnostics: the learned
MoE with randomized-route counterfactual and the same MoE with its router frozen. Both reached
epoch 3 at the fresh signature. Container 2875 GPUs 0/1 were at 98%/99% utilization with
8,071/8,045 MiB allocated, fresh persistent train logs, six output/log files, and zero fatal
matches; W&B runs are `0e0s07va` and `8fxyb9ry`.

Containers 2874, 2862, and 2859 were each inspected separately: all six GPUs report 0% utilization,
0 MiB allocated, and no relevant training process. They remain safely idle because the failed kill
gate forbids replication and no additional diagnosis arm is predeclared; filling them with extra
seeds or speculative sweeps would not change the next decision. The Channel-Adaptive checkpoint is
still absent. OOD test remains untouched.

The incomplete-grid aggregator was also repaired after it emitted an intercept-only
`ccas_effects.csv` from the three-arm kill comparison. Factorial effects are now withheld when no
factor varies; the misleading file is preserved explicitly as
`ccas_effects.excluded_incomplete_seed0.csv`. The valid `ccas_summary.csv`, `ccas_paired.csv`, and
`ccas_report.md` remain. The focused aggregate tests pass 23/23 and the complete SciServer suite
passes 92/92.

Next: strictly validate both diagnosis JSONs, then measure learned-route reliance and whether a
trainable router matters relative to the frozen-router control. This can explain the small recovery
over dense-wide, but it cannot reverse the failed predictive gate or license replication.

## 2026-08-01 04:39 EDT route-audit defect repaired; bounded diagnosis retried once

Classification is `RUNNING_HEALTHY`. The first learned-router and frozen-router diagnosis jobs
completed and produced two parseable, finite, test-blind JSONs. Their performance and randomized-
route fields are diagnostic, but the files are excluded from the final mechanism decision for two
explicit reasons. First, both record `git_dirty=true`: a provenance audit confirmed that the only
modified files were the unrelated aggregation reporter and its test, while every training and
mechanism runtime file still matched HEAD `1344e4d`. Second, and decisively, both JSONs report a
mechanism failure: 10,437,284 token assignments were compared with only 40,612 image-level
experiment/class labels, so routing mutual information and expert usage were not computed.

The excluded attempt is preserved under `kill_rxrx1/native_cp5/diagnose/`. It offers only a
provisional signal: the learned MoE reaches OOD validation `0.114167`, and randomizing its routes
changes accuracy by just `0.000406` (0.04 points); the frozen-router model reaches `0.119850`, 0.57
points above the learned router, and has 0.63-point randomized-route reliance. This pattern argues
against strong learned-route dependence, but it is not a final claim because the mechanism audit
failed and the files lack clean terminal provenance.

The defect was narrow and architectural: token routing emits one assignment per token, while the
audit supplied one site/class label per image. GitHub commit `5874e17` repeats each image label in
image-major token order and checks the router-recorded token count. The repair was transferred to a
new isolated checkout, regression-tested for image and token routing, and frozen as clean SciServer
commit `ac69d40` (tree `01abac6a9e79ad7bb5b7cd8c97c432bbfda4fab4`). Targeted capacity/routing
tests pass 21/21 and the complete remote suite passes 93/93.

The same two predeclared diagnostics—not new arms—were dry-run as exactly two pending results in a
distinct persistent retry root and relaunched once. Container 2875 GPU 0 runs the learned/randomized
route job (W&B `gm70n7gi`) and GPU 1 runs the frozen-router job (W&B `0jceiq4b`). Fresh health proof
shows 98%/97% utilization, 8,021/7,995 MiB allocated, live processes, fresh logs, and zero fatal
matches. Containers 2874, 2862, and 2859 were each inspected separately; all six other GPUs report
0% utilization, 0 MiB, and no relevant process. They remain idle because this single permitted
retry is the only work that can resolve the mechanism decision. OOD test remains untouched.

Next: strictly validate the two retry JSONs from clean `ac69d40`, require the routing error to be
absent and the assignment/label counts to align, then compare learned, randomized, and frozen
routing. Recurrence of the same failure is a blocker; no further retry, replication, or architecture
search is licensed.

## 2026-08-01 05:14 EDT bounded route diagnosis completes cleanly; H2 not supported

Classification is `ACTIONABLE` for milestone persistence and manuscript synchronization. Both
single-retry JSONs completed under clean isolated SciServer commit `ac69d40` (tree
`01abac6a9e79ad7bb5b7cd8c97c432bbfda4fab4`) and pass strict parseability, finiteness,
filename/run/config/seed, parameter, checkpoint/source, four-environment, split, and test-blindness
checks. The repaired audit has no `routing_error`; token-level experiment and class labels are
aligned, all eight experts are observed, and the routing metrics are finite. Both workers exited
normally, container 2875 GPUs 0/1 returned to 0 MiB, and no fatal log match was found. Containers
2874, 2862, and 2859 were each inspected separately and all six remaining GPUs are also idle with
0 MiB and no relevant process.

| Router control | Train | ID | OOD-val | Worst | Randomized-route OOD | Route reliance | Experts | Entropy | MI(experiment) | MI(class) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Learned | 0.386854 | 0.261942 | 0.116501 | 0.014610 | 0.112340 | 0.004161 | 8 | 0.999880 | 0.005104 | 0.001635 |
| Frozen | 0.389343 | 0.265340 | 0.118429 | 0.015016 | 0.111325 | 0.007104 | 8 | 0.997533 | 0.009765 | 0.004069 |

This is the final bounded mechanism diagnosis, not a new efficacy comparison. Randomizing the
learned routes changes OOD-validation accuracy by only 0.42 points, the frozen router is 0.19
points better than the learned router, normalized usage entropy is essentially maximal, and aligned
route mutual information with experiment and class is near zero. H2 is therefore not supported:
the canonical learned router does not discover a reusable batch- or biology-specialized partition
that is necessary for its predictions. A fixed random partition can produce the same or slightly
better behavior, and even that partition has less than one point of route reliance.

The result sharpens, but does not enlarge, the negative seed-0 kill conclusion. It is compatible
with sparse activation modestly regularizing a harmful dense widening, random partitioning creating
weakly useful subspaces, or ordinary single-seed variation. It does not show that learned routing
solves cross-batch gradient interference; H1 was not established. The frozen +5-point gate still
forbids seeds 1/2, a router or architecture grid, additional mechanism arms, and OOD-test access.
OOD test remains untouched.

Next: persist this verified milestone in the paper and synchronized repositories. No additional
GPU experiment is licensed by the completed negative gate.

## 2026-08-01 05:27 EDT verified negative milestone synchronized; kill campaign complete

Classification is `COMPLETE` for the frozen RxRx1 kill-test campaign. GitHub commit `48bf7b9`
contains the two strictly valid clean diagnosis results, final H2 interpretation, updated evidence
maps, and manuscript route-control table. The linked Overleaf project pulled that commit and then
reported no newer GitHub commit since the merge. `paper/main.tex` compiled to six pages with zero
errors and one unchanged underfull-page warning; local fatal-error compilation and rendered-page
inspection also pass.

All licensed work is resolved: native Cell-DINO competence passed; the exact matched seed-0 kill
contrast failed the +5-point replication trigger; the sole bounded learned/randomized/frozen route
diagnosis completed cleanly; seeds 1/2, folds, a grid, deeper mechanism work, and OOD-test access
were never licensed. All eight H100s are idle with no relevant process. The mapping-free
Channel-Adaptive checkpoint remains unavailable but is not required for the completed matched
Cell-DINO decision. Further compute now requires a new scientific question or an explicit protocol
decision from the user.

## 2026-08-01 12:17 EDT substrate-strength hypothesis campaign launched

Classification is `RUNNING_HEALTHY`. The user authorized a new question after the completed
ten-epoch kill campaign: determine whether longer benchmark-strength adaptation and a small set of
mechanistically distinct interventions can expose a stronger Cell-DINO signal before deciding the
fate of sparse capacity. This does not relabel the earlier negative result. The canonical WILDS
ResNet positive control has now completed its 90-epoch trajectory and peaks at OOD-validation
accuracy `0.195149` at epoch 82, with ID accuracy `0.356249` and train accuracy `0.999877`. It is a
diagnostic reference, not a selectable Cell-DINO arm.

The user explicitly rejected a ten-cell optimizer sweep. The replacement is the exact ten-arm,
seed-0, 90-epoch hypothesis matrix frozen in `PLAN.md`: original anchor; matched dense-wide;
canonical token top-1 MoE; image top-1 MoE; within-experiment-balanced token MoE; token top-2 MoE;
frozen linear probe; last-four-block adaptation; experiment-adversarial output invariance; and
environment-balanced classification. Every arm uses the same native CP5 pixels, data order,
optimizer/schedule, and OOD-validation selection policy except for its named intervention. All
runs record metrics at epochs 10/30/60/90; only the original anchor saves checkpoints at those
milestones. This prevents four duplicate epoch-budget runs and spends the parallelism on competing
technical explanations.

The execution implementation is GitHub `d4e80b0`. Its code-equivalent clean SciServer checkout is
commit `cd783399ab1d4cee2666f1af8dfe3bfd9fc29280`, tree
`ec05b37b9f2ac593e047243c414b114c3a1fb52c`; five focused tests and the complete 99-test remote
suite pass. Five disjoint two-GPU shards launched exactly two arms apiece under the persistent root
`substrate_rxrx1/cell_dino_cp5/hypothesis90/` and launcher logs
`logs/hypothesis90_20260801/shard{0..4}.launcher.log`. The resource map is:

- host `2b741f13...`: original anchor and token top-2 MoE;
- host `8cf2b351...`: matched dense-wide and frozen linear probe;
- host `88e600c3...`: canonical token top-1 MoE and last-four-block adaptation;
- host `d644003b...`: image top-1 MoE and output-invariant adaptation;
- host `a70b09cf...`: within-experiment-balanced token MoE and environment-balanced adaptation.

All ten direct workers remain alive after more than five minutes, all ten H100s hold the expected
model memory, and every arm has written an epoch-1 training record. Per-shard fatal scans are clear.
Every worker reports the same 33 training experiments and exact selection coverage: train `40,612`,
ID-test `40,612`, OOD-validation `9,854`; OOD test `34,432` is recorded as untouched and is not
evaluated. Therefore current coverage is `0/10` strictly valid final results, `10/10` active, and
zero queued or idle H100s. Epoch-1 losses are optimization-health evidence only and are not
interpreted scientifically.

All runs are isolated in the fresh W&B group `rxrx1-cell-dino-hypothesis90-20260801`, with job type
`rxrx1_hypothesis_matrix` and test-blind tags. The ten W&B run IDs are `6zkqodfv`, `0xpwd8mc`,
`jo0b8ycc`, `w6sqdgov`, `0crn85p0`, `j9yjhagx`, `1zwn6qq5`, `ixgx0uhr`, `rmsua36b`, and `kmotbuzn`,
mapped one-to-one to the arm identities in their persistent logs. Strict Hugging Face publishing is
configured under `rxrx1/cell_dino_cp5/hypothesis90_20260801`; no folder has been claimed uploaded
yet. A run folder is pushed only after the final JSON and all milestone/checkpoint artifacts pass
the exact identity, finite-metric, OOD-test-blindness, and SHA-256 manifest checks.

The 15-minute `moe-sciserver-progress-check` heartbeat is active with the new campaign contract.
It must inspect all five containers independently, validate and publish completed folders, and may
advance only through the milestone and epoch-90 gates above; it is explicitly forbidden from
turning this batch into a hyperparameter sweep.

Next: preserve all ten healthy workers and validate the first 10-epoch milestone files as they
appear. Those paired train/ID/OOD curves decide whether the former ten-epoch contrast was simply
undertrained; the 30/60/90 trajectory then separates capacity, routing granularity, routing
starvation, representation depth, explicit invariance, and environment imbalance. Only a valid
epoch-90 MoE-minus-dense-wide gain of at least five OOD points with at most two ID points lost can
license replication. OOD test remains sealed throughout.

## 2026-08-01 13:07 EDT epoch-10 milestone validated; sixth container queued

Classification is `RUNNING_HEALTHY`. All ten hypothesis90 arms produced exactly one epoch-10
milestone row. The rows are parseable, finite, exact seed/run/epoch matches, use
`selection_split=ood_val`, set `test_evaluated=false`, and contain no exclusion. OOD test remains
untouched. No final result JSON exists, so final-result coverage remains `0/10`; milestone coverage
is now `10/10` at epoch 10. The original anchor checkpoint exists at the declared persistent path,
is `269,082,312` bytes, and has SHA-256
`d175761bce0444b6845a75b69410481d2f6f647bf73ff3a8b60642bbcdb5361`.

The paired interim metrics are:

| Arm | Train | ID | OOD-val | Worst OOD environment |
|---|---:|---:|---:|---:|
| original anchor | 0.524793 | 0.335369 | 0.137711 | 0.014610 |
| matched dense-wide | 0.476390 | 0.295504 | 0.133144 | 0.012987 |
| canonical token top-1 MoE | 0.473778 | 0.300970 | 0.128983 | 0.015422 |
| image top-1 MoE | 0.459730 | 0.288757 | 0.128679 | 0.011364 |
| within-experiment-balanced token MoE | 0.474616 | 0.301438 | 0.128374 | 0.015828 |
| token top-2 MoE | 0.517892 | 0.315892 | 0.136290 | 0.014610 |
| frozen linear | 0.095303 | 0.062642 | 0.033590 | 0.007711 |
| last-four-block adaptation | 0.527775 | 0.345120 | 0.127765 | 0.016640 |
| output invariance | 0.476513 | 0.312814 | 0.125837 | 0.012581 |
| environment-balanced classification | 0.363836 | 0.252659 | 0.115182 | 0.012987 |

This is diagnostic and provisional. Canonical top-1 MoE is `0.416` percentage points below
dense-wide on OOD validation, while top-2 is `0.315` points above dense-wide but uses unmatched
active compute and remains `0.142` points below original. The original anchor therefore still leads
the fair efficacy family at this milestone. Frozen linear remains near floor, showing that the
checkpoint is not immediately linearly separable for this task. Last-four adaptation has the best
ID accuracy but not OOD accuracy, which is compatible with representation-depth effects that do
not yet improve transfer. These trajectories could still change materially by epochs 30/60/90;
they do not pass or fail the frozen epoch-90 gate.

All five assigned containers were inspected separately through fresh terminal signatures. Their
ten direct workers are alive and mapped to the expected persistent logs; GPU memory is allocated
to every arm, fatal scans are clear, and the latest recorded epochs range from 14 to 19. Host
`88e600c3...` returned three zero-utilization samples, but both processes consumed CPU and their
logs advanced through epoch 19 within seconds, so the shard is healthy rather than stalled. The
other hosts showed active GPU samples up to 100%. Portal status for all five is `Running`.

The user-authorized sixth 2-H100 container was created as `tester6` (container `2893`, external ref
`8bf4e9aa-8dc8-11f1-a24e-0a580a8201b9`) with the same persistent/scratch/data mounts. Its start
request reached the scheduler, but the authoritative container state remains `Pending` because no
eligible H100 pair is currently available: three nodes report insufficient GPU capacity, two are
unschedulable, and the rest fail the node selector. No duplicate container or repeated start was
issued. The 15-minute steward now checks this exact sixth container every invocation, recreates it
only if absent, and assigns work only when a scientifically licensed independent arm exists.

W&B identities and the fresh campaign group remain recorded in logs and state, but a direct API
status query returned `relogin required`; no live W&B state is claimed from that failed query. No
Hugging Face folder is published because no run is complete. The exact milestone metrics, paths,
container signatures, checkpoint digest, tracking limitation, and tester6 scheduler state are
preserved in `analysis/hypothesis90_epoch10_validation.json`.

Next: protect the ten healthy workers and validate the 10 paired epoch-30 milestone rows. That is
the next point at which the trajectories can distinguish transient undertraining from a stable
original/dense-wide/MoE ordering. If tester6 becomes `Running` first, verify both GPUs and use it
only for already licensed decision-relevant work; do not invent a filler sweep. OOD test remains
sealed.

### 2026-08-01 13:46 ET — W&B provenance repaired; epoch-30 gate is 9/10

A signed W&B table audit corrected one transcription defect in the campaign provenance: the
canonical token-top-1 run ID is `jo0b8ycc` (letter `o`, then zero), not `j0b8ycc`. The corrected
identity is now consistent across the operational ledger, evidence index, machine state, and the
epoch-10 validation artifact. No metric or scientific interpretation changed.

Fresh W&B run pages show all ten hypothesis90 runs in `Running` state with no rendered fatal
trace. Nine arms have reached or passed epoch 30: dense-wide `30`, image-top-1 `34`, canonical
token-top-1 `34`, token-top-2 `33`, environment-balanced `34`, frozen-linear `34`, original
anchor `33`, output-invariant `33`, and last-four-block `33`. The within-experiment-balanced token
arm is the remaining straggler at epoch `27`, with a fresh log. Therefore the epoch-30 scientific
gate remains incomplete and no cross-arm epoch-30 claim is made yet. SciServer `tester6` still
returns `503` at its execution endpoint, consistent with the previously recorded pending/no-H100
state; no duplicate or repeated start request was issued. OOD test remains sealed.

### 2026-08-01 14:01 ET — epoch-30 W&B coverage complete; strict gate blocked on SciServer login

All ten signed W&B run logs now contain an epoch-30 milestone and remain `Running` with no rendered
fatal trace. The rounded operational snapshot is preserved in
`analysis/hypothesis90_epoch30_wandb_snapshot.json`. Canonical token top-1 is provisionally
`+2.78` OOD-validation points and `+5.27` ID points versus exact-total-parameter-matched dense-wide;
image top-1 is `+3.06` OOD points, within-experiment-balanced token is `+2.41`, and top-2 is
`+1.22`. Original is `+2.23` over dense-wide. These rounded W&B lines suggest the early sparse
ordering improved after epoch 10, but none reaches the frozen `+5`-point replication trigger and
none is yet a validated result.

Strict epoch-30 validation is blocked because the SciServer portal session now requires
reauthentication, so the persistent milestone JSONs cannot be checked for exact precision,
parseability, identity, parameter counts, `selection_split=ood_val`, `test_evaluated=false`, and
null OOD-test fields. W&B cannot substitute for those checks. The ten healthy workers continue
untouched toward epoch 60. `tester6` still returns `503`; no duplicate or repeated start was issued.

### 2026-08-01 15:31 ET — epoch-60 W&B coverage complete; moderate signal remains unvalidated

All ten signed W&B logs contain epoch-60 milestones and remain `Running` with fresh post-milestone
progress and no rendered fatal trace. The rounded operational snapshot is preserved in
`analysis/hypothesis90_epoch60_wandb_snapshot.json`. Canonical token top-1 is provisionally
`19.68%` OOD validation versus `17.13%` for exact-total-parameter-matched dense-wide, a `+2.55`
point gain, while ID is `47.16%` versus `43.61%`. The gain is below the frozen `+5`-point
replication trigger. Worst-experiment accuracy is nearly unchanged (`1.54%` versus `1.50%`).

The nearest alternatives are within-experiment-balanced routing at `19.43%` OOD validation and
last-four-block adaptation at `19.26%`; original is `18.51%`, image top-1 `18.73%`, and top-2
`17.97%`. Thus the average OOD signal is persistent but not uniquely attributable to sparse
routing, and it has not translated into a worst-experiment improvement. Frozen linear remains
weak at `5.91%` OOD validation. This is a complete operational milestone, not scientific evidence:
all values are rounded W&B renderings and the epoch-90 gate remains controlling.

Strict validation is still blocked on SciServer portal reauthentication, including exact JSON
precision, identity, parameter counts, split/test-blindness fields, and the epoch-60 anchor
checkpoint. The workers continue untouched toward epoch 90. `tester6` still returns `503`; no
duplicate or repeated start was issued, and OOD test remains sealed.

### 2026-08-01 16:17 ET — eight final W&B handoffs; two arms still running

Signed W&B pages now mark eight arms `Finished` with epoch-90 milestones: image top-1, canonical
token top-1, token top-2, environment-balanced classification, frozen linear, original anchor,
output invariance, and last-four-block adaptation. Matched dense-wide remains healthy at epoch 87
and within-experiment-balanced token routing remains healthy at epoch 73. Both have fresh logs and
no rendered fatal trace. This is `8/10` operational final coverage, not a completed comparison;
no epoch-90 metric is interpreted or used for the gate.

The eight completed workers release capacity inside their existing containers, but no new work is
licensed before the full epoch-90 gate and the remaining exact validation cannot proceed while the
SciServer session is unauthenticated. No speculative sweep or duplicate was launched. `tester6`
still returns `503`, OOD test remains sealed, and no Hugging Face upload is claimed because the
finished result folders have not passed strict persistent-file validation.

### 2026-08-01 16:32 ET — matched dense-wide finished; primary pair operationally below trigger

Matched dense-wide has transitioned to `Finished`, bringing operational final coverage to `9/10`.
Within-experiment-balanced token routing is the sole remaining arm, healthy at epoch 79 with a
fresh fatal-free log. The rounded W&B primary-pair snapshot is preserved in
`analysis/hypothesis90_final_wandb_snapshot.json` but is explicitly barred from formal gate use.

Canonical token top-1 is `20.22%` OOD validation versus `18.50%` for exact-total-parameter-matched
dense-wide: `+1.72` points, well below the frozen `+5` trigger. ID is higher by `+3.22` points, but
worst-experiment accuracy is `0.08` points lower. Original (`20.09%`), image top-1 (`20.13%`),
environment balancing (`19.99%`), and last-four adaptation (`19.82%`) are all close, so the rounded
ordering does not establish a sparse-routing-specific robustness effect. Because one arm is still
running and exact persistent files remain inaccessible, this is operational evidence only—not a
gate decision, exclusion, or paper claim.

Nine allocated worker GPUs are now idle inside the existing containers, but replication is not
licensed by the rounded result and no speculative work was launched. SciServer reauthentication
is still required for strict result/checkpoint validation and Hugging Face publication. `tester6`
continues to return `503`; OOD test remains sealed.

### 2026-08-01 17:02 ET — hypothesis90 matrix 10/10 finished operationally; strict gate blocked

Within-experiment-balanced token routing finished cleanly, so all ten W&B runs now report
`Finished`, each has an epoch-90 milestone, and no rendered fatal trace appears. The completed
rounded snapshot is in `analysis/hypothesis90_final_wandb_snapshot.json`. No worker remains active;
the ten allocated H100s are idle, but the predeclared performance threshold does not license
replication or mechanism work and no speculative launch was made.

The controlling canonical comparison remains token top-1 `20.22%` versus exact-total-parameter-
matched dense-wide `18.50%`: `+1.72` OOD-validation points, below the `+5` replication trigger,
with ID `+3.22` but worst-experiment `-0.08` points. Within-experiment routing is the highest
average OOD arm at `20.46%`, but it is only `+1.96` over dense-wide and its worst-experiment score
is `1.38%`, substantially below dense-wide's `1.70%`. Original (`20.09%`) and image top-1
(`20.13%`) remain effectively tied with canonical routing. Operationally, longer training rescued
the substrate but did not produce a large or tail-robust sparse advantage.

This is still not the formal gate decision. SciServer reauthentication is required to validate all
ten exact result JSONs, milestones, parameter counts, `selection_split=ood_val`,
`test_evaluated=false`, null OOD-test fields, logs, and anchor checkpoints; only then can manifests
be created and valid folders uploaded to Hugging Face. `tester6` remains unavailable with `503`.
OOD test remains sealed, and there are zero scientific exclusions.

### 2026-08-01 20:12 ET — hypothesis90 strictly validated; formal negative gate

Classification is `ACTIONABLE` for the final source-of-truth and manuscript handoff. SciServer
reauthentication restored direct access to the persistent campaign. All ten final result JSONs,
all 40 milestone rows, the four declared original-anchor checkpoints, all ten SHA-256 manifests,
and the execution logs now pass strict validation. The exact execution identity is commit
`cd783399ab1d4cee2666f1af8dfe3bfd9fc29280`, tree
`ec05b37b9f2ac593e047243c414b114c3a1fb52c`, Cell-DINO checkpoint SHA-256
`37d20e9cd48b3d610b5de15a4ea4e7e060a593b8d8358e928d079dc7b03ee66a`, and DINOv2
commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`. Every result is seed 0, uses
`selection_split=ood_val`, records `test_evaluated=false`, and has null OOD-test metrics. There
are no scientific exclusions.

The strict audit found one uniform metadata-only defect: the two per-environment OOD-test maps
were empty dictionaries rather than explicit JSON nulls. No OOD-test value existed or was
accessed. A narrow repair normalizes all withheld-test fields to null and strengthens the
publisher guard; five focused and 101 full tests pass in isolated repair commit
`84185df6c7356de6d63d49fae13355e4e6621a9`, tree
`9772f6d4a1400812ca81463e40a253976278eb6c`. Original metadata is preserved under
`metadata_pre_null_normalization_20260801/`. The ten corrected result files and manifests were
re-uploaded and force-downloaded from Hugging Face for digest verification; the published prefix
contains 44 files.

The formal primary result is canonical token top-1 MoE OOD validation `0.2021514106` versus
exact-total-parameter-matched dense-wide `0.1850010148`, a gain of `0.0171503958` (1.715 points).
ID improves by `0.0321333596`, but worst-experiment accuracy changes by `-0.0008116883`. The
models differ by only 378 total parameters, or `0.001232%`, within the frozen 0.1% tolerance.
This fails the predeclared `+5`-point replication trigger. The smaller original reaches
`0.2009336310` OOD validation, only 0.122 points below canonical MoE, with a better worst-
experiment score. Within-experiment-balanced routing is the highest average OOD arm at
`0.2045869698`, but its worst-experiment accuracy is 0.325 points below dense-wide. Top-2,
image routing, partial adaptation, output invariance, and environment balancing do not create a
large or tail-robust effect.

This is decision-grade evidence for the bounded seed-0 question. Longer adaptation establishes a
strong Cell-DINO instrument and a genuine ID-to-OOD transfer failure: all adapted arms reach
100% train accuracy and roughly 48--53% ID accuracy, while OOD validation remains roughly
18.5--20.5%. It does not support a large, uniquely sparse, or worst-experiment-robust advantage.
Single-seed variation or a smaller true effect remains possible, but the predeclared target is
falsified by the exact matched contrast. Seeds 1 and 2, mechanism expansion, and OOD-test access
are not licensed.

The exact 10/30/60/90 primary OOD trajectories are original
`0.137711/0.175157/0.185102/0.200934`, dense-wide
`0.133144/0.152933/0.171301/0.185001`, and canonical MoE
`0.128983/0.180739/0.196773/0.202151`. Thus conditional gain is
`-0.004161/+0.027806/+0.025472/+0.017150`: it peaks by epoch 30 and contracts thereafter. Sixty
epochs would have been sufficient for the qualitative negative gate; 90 epochs are still valuable
for the final result because they improve absolute baselines and rule out undertraining. Only the
original anchor has reloadable checkpoints at 10/30/60/90; all arms retain exact milestone
metrics. Future bounded scouting should use 60 epochs, reserving 90 for final adjudication.

Traceable decision artifact: `analysis/hypothesis90_final_validation.json`. W&B group:
`rxrx1-cell-dino-hypothesis90-20260801`. Hugging Face prefix:
`rxrx1/cell_dino_cp5/hypothesis90_20260801`. All ten workers are finished; 10 assigned H100s are
idle. A single portal start request was issued for existing `tester6` container 2893 after login;
the refreshed authoritative portal state remains `stopped`, so the request did not acquire a
two-H100 allocation. No second start request, duplicate container, or speculative GPU work was
launched.

Next: finish the manuscript compile, GitHub/source-of-truth commit, and Overleaf sync for this
validated milestone. No new GPU experiment is licensed by the negative gate.

### 2026-08-01 20:24 ET — validated milestone synchronized and campaign complete

GitHub commit `ae27241addbcd65e8f698b73ee2adea4ba1d19c4` contains the strict final
validation artifact, exact 10/30/60/90 trajectory, withheld-test metadata guard and regression
tests, evidence ledgers, abstract, result table, and duration interpretation. It is pushed to
`origin/main`. The linked Overleaf project imported that exact commit, then compiled the synced
seven-page manuscript with 0 errors and 1 benign `Command \\showhyphens has changed` warning. A
local fatal-error build and complete rendered-page inspection also pass.

Classification is `COMPLETE` for the bounded hypothesis90 campaign. The substrate-strength
question is resolved, all 10 result folders are validated and published, no H100 worker is active,
and the frozen replication threshold is not met. No seed replication, mechanism expansion,
architecture sweep, or OOD-test evaluation is licensed. A new campaign requires an explicit new
scientific question rather than reuse of the failed gate.

### 2026-08-01 21:00 ET — user authorizes Cell-DINO factorial60 sweep; launch tree prepared

Classification is `ACTIONABLE`. The user explicitly opened a new exploratory question rather than
reinterpreting the negative `hypothesis90` gate: test whether placement, routing granularity,
router geometry, or environment pressure exposes a materially stronger sparse effect on the
validated native-CP5 Cell-DINO substrate. `PLAN.md` now predeclares the complete seed-0 43-cell
factorial: 36 MoE arms, six placement/pressure-matched dense-wide controls, and one original
reference, all under the same 60-epoch schedule with train/ID/OOD-validation/worst-experiment
milestones at 10/30/60. OOD test remains sealed. A winner remains exploratory until its exact
configuration is frozen and replicated with paired seeds 1 and 2.

The idempotent launcher `scripts/sweep_rxrx1_cell_dino.py` and focused regression coverage are
implemented locally. It maintains five disjoint shards, continuously refills two GPUs per shard,
uses a clean W&B group `rxrx1-cell-dino-factorial60-20260801`, and defers Hugging Face upload until
strict validation so network transfer does not hold a GPU lease. Every MoE arm saves its epoch-60
checkpoint; original and all six dense comparators save 10/30/60 anchors. Local bytecode and diff
checks pass. The local machine lacks the project test dependencies, so the focused and full pytest
suites remain a mandatory remote launch gate rather than being falsely claimed locally.

Portal inspection shows five existing 2-H100 containers running and `tester6` container 2893 in
authoritative `Pending` state after exactly one fresh start request. Its scheduler reason is
`3 Insufficient nvidia.com/gpu`, with two nodes unschedulable and the remaining nodes failing the
required selector; no duplicate start or container was issued. The five running containers expose
ten H100 allocations, but their job-level utilization still requires fresh per-container terminal
inspection before launch. No sweep worker is yet claimed active.

Next: commit and push the clean campaign tree, run focused and full tests in the pinned SciServer
environment, dry-run all five shards, verify every GPU is free and no duplicate result/process
exists, then launch the ten initial arms. A completion or pruned milestone immediately refills its
GPU from the remaining 33 arms.

### 2026-08-01 21:19 ET — factorial60 sweep launched on all ten available H100s

Classification is `RUNNING_HEALTHY`. GitHub commit `bd213dbd7758f456eb822379707627b1998847ff`
was transferred as a code archive into the isolated SciServer execution checkout, whose local
execution commit is `b8ece25e05dc675bd6a61e0728879e53130e453e`. Both resolve to the exact tree
`7320d64944c34c9ee832924ee06429491739354f`. In the pinned runtime, 30 focused tests and 106
full tests pass. Five dry-run shards are disjoint and exhaustive at 9/9/9/8/8 cells; the duplicate
scan found zero factorial results or workers and each of the ten assigned GPUs was free before
launch.

Five continuous-refill launchers now run two workers each across the five existing 2-H100
containers. The initial ten distinct arms all have fresh epoch-0 train records, approximately
8 GiB allocated on each GPU, and no fatal-error signature. Coverage is 0/43 valid results, with
10 active and 33 queued. The child environments identify the fresh W&B project/group/job type and
tags, and ten live W&B run directories exist under the configured persistent tracking root. HF
publication remains deliberately deferred until each completed folder passes strict result,
milestone, log, checkpoint, split, provenance, and SHA-256 validation; upload must not hold a GPU
lease.

No performance conclusion is available from epoch-0 losses. The active factorial is designed to
separate placement, image-versus-token routing, cosine-versus-linear routers, and environment
pressure, always against an explicit placement/pressure-matched dense comparator. OOD test remains
sealed. Multiple-search optimism is the largest scientific threat, so any apparent seed-0 winner
must pass the frozen effect/ID/tail rule and then be replicated unchanged at seeds 1 and 2.

`tester6` container 2893 remains `Pending` after the single authorized start request because no
eligible two-H100 node is schedulable; no duplicate request or container was created. The next
automatic action is the first completion or 10-epoch handoff: strictly validate the milestone,
apply the predeclared prune/continue rule, refill the freed GPU from the 33-cell queue, and publish
only validated completed folders. This directly shortens the critical path while preserving
OOD-test blindness and comparator fairness.

### 2026-08-01 22:02 ET — first factorial60 epoch-10 handoff validated, pruned, and refilled

Classification is `RUNNING_HEALTHY`. All ten initial MoE cells emitted exactly one epoch-10 row,
and all ten rows pass parseability, exact run/seed/epoch identity, finite train/ID/OOD-validation/
worst-experiment metrics, four-environment coverage, fixed per-environment counts,
`selection_split=ood_val`, `test_evaluated=false`, ERM-objective, tracking-identity, and fatal-log
checks. The source/execution/tree remain `bd213dbd7758f456eb822379707627b1998847ff` /
`b8ece25e05dc675bd6a61e0728879e53130e453e` /
`7320d64944c34c9ee832924ee06429491739354f`. Exact normalized rows are in
`analysis/factorial60_epoch10_validation.json`; OOD test remains sealed.

Early token-cosine canonical leads mean OOD validation at `0.134869`, while early image-linear
route is second at `0.133245` and has the stronger worst-experiment score (`0.018263`). This is a
diagnostic ranking across searched MoE cells, not sparse-efficacy evidence: the paired dense
controls have not yet reached the same milestone. Search multiplicity and the absent matched
contrast are the largest threats.

The predeclared strict Pareto rule on OOD validation, ID retention, and worst-experiment accuracy
pruned four dominated cells: early image-linear output, early image-cosine route, early token-linear
route, and early token-linear canonical. Their epoch-10 rows and logs are preserved. Six frontier
cells continue. The four released leases immediately refilled with early token-cosine output,
middle image-linear canonical, middle image-cosine output, and middle image-linear output. All four
new workers have fresh W&B directories in the declared group, correct job/tags, model allocations,
and no fatal trace. Coverage is 10/10 validated diagnostic milestones for the first wave, 0/43
completed valid results, 4 pruned, 10 active, 29 queued, and 0 idle among the ten allocated H100s.

`tester6` remains authoritatively `Pending`: 3 nodes lack the requested GPUs, 2 are unschedulable,
and 11 fail affinity; the single prior start request was not repeated. Next: validate the next
epoch-10 handoff, apply the same declared pruning rule, and immediately refill every released GPU.
The first MoE-minus-dense efficacy comparison is licensed only after the exact placement/pressure-
matched dense control reaches the same milestone.

### 2026-08-01 23:08 ET — second epoch-10 wave and first epoch-30 wave validated; eight leases refilled

Classification is `RUNNING_HEALTHY`. Four additional epoch-10 rows and all six epoch-30 rows from
the first frontier pass exact run/config/seed/epoch identity, finite train/ID/OOD-validation/worst-
experiment metrics, four-environment coverage and counts, ERM objective, `selection_split=ood_val`,
`test_evaluated=false`, absent OOD-test fields, shared commit/tree provenance, tracking identity,
and fatal-log checks. Cumulative validated coverage is 14 epoch-10 milestones and six epoch-30
milestones, with 0/43 completed result JSONs. Exact normalized evidence is in
`analysis/factorial60_wave2_validation.json`.

At epoch 30, early image-linear route reaches `0.200223` OOD validation, `0.475672` ID, and
`0.019075` worst-experiment accuracy and strictly dominates the other five first-wave survivors
on all three triage axes. It advances to epoch 60. Middle image-linear output (`0.129186` OOD,
`0.308923` ID, `0.018263` worst experiment) is the only cell from the new epoch-10 wave that
remains on the cumulative frontier and advances to epoch 30. Eight dominated cells were stopped
only after their complete validated milestones; all rows and logs remain preserved. These are
exploratory allocation decisions, not sparse-efficacy evidence.

All eight released GPUs immediately refilled with distinct next cells, restoring ten direct
workers across containers 2887/2875/2874/2862/2859. Each container owns two workers; current
per-GPU model allocations are about 8.0--8.1 GiB, all worker environments report the declared W&B
group/job/tags, the five fatal scans are empty, and the execution checkout remains clean at
`b8ece25e05dc675bd6a61e0728879e53130e453e`. Instantaneous GPU utilization varies during CPU-heavy
evaluation/data phases but no allocated GPU is idle: all ten direct workers are alive with fresh
logs. Current accounting is 10 active, 21 queued, 12 cumulatively pruned, and 0 idle among the ten
available H100s; 22 W&B runs have been launched cumulatively. HF publication remains pending
strict completed-result validation.

The dense baselines are known, not missing. The six controls are early/middle/late crossed with
canonical/output pressure; route-balanced MoE pairs with same-placement canonical dense. The
current leader's exact comparator is
`rxrx1_dense_wide_early_canonical_E8_ep60_s0_factorial60_20260801`. Its new 60-epoch-schedule
milestone has not emitted yet, so no conditional-gain claim is made; the older validated middle-
canonical 90-epoch dense trajectory remains context but is not the exact placement/schedule pair.
This comparator is the highest-priority safe refill once a lease can be reassigned without
interrupting a healthy pre-milestone worker. Search multiplicity and the pending exact paired
contrast remain the largest threats.

`tester6` container 2893 remains authoritatively `Pending`: 11 nodes fail affinity, two are
unschedulable, and three lack the requested GPUs; the prior single start request was not repeated.
Next: strictly validate the next complete epoch-10 wave, apply the frozen Pareto rule, promote the
early canonical dense comparator at the next safe refill opportunity, and continue the two current
frontier arms to their licensed milestones. OOD test remains sealed.

## 2026-08-01 23:26 EDT factorial60 wave-three handoff

Eight additional epoch-10 rows are strictly valid, raising cumulative factorial60 coverage to 22
epoch-10 and six epoch-30 milestones; no final result JSON has completed. Early token-cosine route
is the sole new Pareto survivor and leads cumulative epoch-10 mean OOD validation and ID accuracy
at `0.136391` and `0.317615`, with worst-experiment accuracy `0.013393`. The other seven cells are
strictly dominated by the already validated early token-cosine canonical cell on all three frozen
triage axes and were stopped only after their complete milestones were preserved.

All seven released leases refilled immediately. Ten direct workers remain active across containers
2887/2875/2874/2862/2859 and 14 arms remain queued; cumulative pruning is 19 and 29 W&B runs have
launched. The seven new arms are middle token-linear output, middle token-cosine canonical, late
token-linear output, early output-pressure dense-wide, middle token-cosine output, late image-cosine
route, and late image-linear canonical. Direct process ownership, two model allocations per
container, and the declared W&B group/job/OOD-test-blind tags were verified. The three protected
survivors remained live. No available H100 is idle.

This is an exploratory allocation result, not evidence that sparse computation beats dense. The
early canonical dense comparator is known and still queued; the early output-pressure dense control
has now launched for its own exact output-pressure contrasts. The sharp falsifier is persistence of
the early token-cosine route lead at epoch 30 followed by at least +5 absolute OOD-validation points
over its exact early canonical dense control with no more than two ID points lost, or the separately
predeclared consistent tail improvement. Search multiplicity is the largest threat. `tester6`
container 2893 remains Pending for 11 affinity mismatches, two unschedulable nodes, and three nodes
without requested GPUs; no duplicate request was issued. OOD test remains sealed.

## 2026-08-01 23:40 EDT — literature-backed architectural queue predeclared during healthy execution

Classification remains `RUNNING_HEALTHY`. A fresh audit of containers 2887/2875/2874/2862/2859
finds the same ten direct owned factorial60 workers and two approximately 8 GiB model allocations
per container. Cumulative scientific coverage is unchanged at 22 valid epoch-10 and six valid
epoch-30 milestones, with no final result JSON, no new checkpoint, and no fatal file. All ten
available H100s remain assigned; no worker was interrupted. `tester6` 2893 is still `Pending`
because 11 nodes fail affinity, two are unschedulable, and three lack the requested GPUs;
preemption is unhelpful and the single prior start request was not repeated.

The active tested refill queue remains 14 factorial60 cells. To keep the next architectural family
ahead of lease turnover, `analysis/architectural_hypothesis_backlog.md` now predeclares 24 bounded,
literature-backed questions spanning sparse upcycling, staged router/expert unfreezing, router
z-loss and balance strength, expert-choice and soft routing, shared experts, well-level routing,
augmentation consistency, representation preservation, gradient-conflict-localized placement,
tail-safe robust objectives, route mechanism controls, expert count, capacity, and temperature.
Each row records an exact comparator, predicted signature, alternative explanation, falsifier,
fairness class, horizon, and implementation gate. The top 12 next-family designs are designated for
isolated implementation and testing; they are not called runnable until those gates pass.

This action creates no performance claim and does not alter factorial60's frozen pruning or
replication rules. The immediate automatic action remains strict validation of the next complete
milestone and refill from the existing tested queue, with the known early-canonical dense control
highest priority. In parallel, implementation work begins with the first 12 next-family designs so
the next sweep can start without leaving a future lease idle. OOD test remains sealed; multiplicity
and failure to separate sparse routing from objective or active-compute effects remain the largest
scientific threats.

## 2026-08-02 00:11 EDT factorial60 wave-four and first completion

Nine new milestone rows are strictly valid: seven epoch-10, one epoch-30, and the first epoch-60
row. Cumulative coverage is now 29/7/1 at epochs 10/30/60, with one validated final result. The
completed early image-linear route arm reaches train/ID/OOD-validation/worst-experiment accuracy
`1.000000/0.527529/0.211082/0.024756`. From epoch 30 to 60 it improves OOD validation by `0.010859`,
ID by `0.051857`, and worst experiment by `0.005682`, so the directional signal is not merely an
epoch-30 spike. This remains exploratory: its exact same-placement canonical dense comparator is
known but has not emitted.

The new early output-pressure dense-wide control is the strongest epoch-10 mean-OOD cell so far at
`0.144510`, ahead of every searched MoE epoch-10 row. Middle token-cosine canonical remains on the
tail frontier at OOD/ID/worst `0.131520/0.303162/0.019481`. Five other new epoch-10 rows and the
middle image-linear output epoch-30 row are strictly dominated on all frozen triage axes and were
stopped only after validation. This strengthens ordinary capacity/pressure effects as an
alternative to conditional routing; it is not yet the exact route-pressure comparison.

The completed result JSON, three milestones, 60-epoch checkpoint, manifest, logs, finite metrics,
four validation environments/counts, clean `b8ece25e`, `selection_split=ood_val`,
`test_evaluated=false`, and null held-out-test metrics validate. The 368,488,165-byte checkpoint has
SHA-256 `059df6d74e8376930066120cc5ccfe698329c1d1bd8eab74504e112fd8d42fea`.
Five files were uploaded successfully and re-listed under the declared factorial60 HF prefix.

Six dominated live workers plus the completed worker released seven leases. All seven refilled
immediately with late image-linear route, late image-linear output, original, late token-cosine
canonical, middle canonical dense-wide, late image-cosine output, and late token-cosine route.
Ten workers remain active, seven tested cells remain queued, cumulative pruning is 25, 36 W&B runs
have launched, and none of the ten available H100s is unassigned. `tester6` 2893 remains Pending:
11 nodes fail affinity, two are unschedulable, and three lack GPUs; the single start request was not
repeated. OOD test remains sealed. The next gate is the exact early-canonical dense milestone and
the next validated halving/refill; multiplicity and seed-0 winner selection remain the largest
threats.

## 2026-08-02 00:29 EDT factorial60 late epoch-30 frontier handoff

A late-arriving epoch-30 row from the protected early token-cosine route-pressure arm is strictly
valid, raising cumulative factorial60 coverage to 29/8/1 at epochs 10/30/60. It reaches
train/ID/OOD-validation/worst-experiment accuracy
`0.999906/0.467596/0.194134/0.021510`. Relative to its own epoch-10 row, OOD validation improves
`0.057743`, ID `0.149980`, and worst experiment `0.008117`; the signal is not an early transient.

Against the otherwise matched early token-cosine canonical MoE at epoch 30, route pressure changes
OOD validation by `+0.004364`, ID by `-0.006993`, and worst experiment by `+0.006088`. It is now the
second-best validated epoch-30 mean-OOD cell and the epoch-30 worst-experiment frontier. This is an
exploratory routing-pressure mechanism contrast, not sparse-versus-dense efficacy. The exact
early-canonical dense control remains known and queued.

The JSONL has unique epoch-10/30 identities, finite metrics, four expected validation environments,
ERM, `selection_split=ood_val`, `test_evaluated=false`, and no held-out-test metric. The active
worker still matches the frozen seed-0 early/token/cosine/route/E8/top-1 configuration and has
advanced past epoch 37 without a fatal log. It remains licensed to epoch 60 under the frozen Pareto
rule. All ten available H100s remain assigned to ten distinct workers; seven tested arms remain
queued. `tester6` 2893 is still Pending for the unchanged 11-affinity, 2-unschedulable,
3-insufficient-GPU reason. OOD test remains sealed.

## 2026-08-02 00:38 EDT factorial60 epoch-10 prune and refill

Late image-linear output routing is strictly valid at epoch 10 but is dominated on OOD validation,
ID, and worst-experiment accuracy: `0.119444/0.293386/0.015422`. The already validated middle
token-cosine canonical cell exceeds it by `0.012076/0.009775/0.004058` on those axes. The worker was
terminated only after its complete row was preserved, and shard 1 immediately refilled GPU 0 with
the distinct tested late token-linear route arm. Cumulative coverage is 30/8/1, pruning is 26,
37 W&B runs have launched, 10 workers are active, six tested arms remain queued, and zero available
H100s are unassigned. All split/test guards remain intact; tester6 remains Pending for the recorded
capacity reason.

## 2026-08-02 00:59 EDT factorial60 wave-seven handoff and complete launch coverage

Eight new milestones pass strict JSONL, exact run/seed/epoch/config, finite-metric,
four-environment/count, ERM, tracking, fatal-scan, `selection_split=ood_val`, and
`test_evaluated=false` checks. Six are epoch 10 and two are epoch 30, raising cumulative coverage to
36/10/1 with one validated final result. Original Cell-DINO is Pareto-relevant at epoch 10 with
OOD/ID/worst `0.137609/0.335664/0.014205`. Late image-linear route also continues because its
`0.131723/0.293632/0.014610` tradeoff is not strictly dominated.

Six rows are strictly dominated and were stopped after preservation: dense early output at epoch
30; late token-cosine route, middle canonical dense, late token-cosine canonical, and late
image-cosine output at epoch 10; and middle token-cosine canonical at epoch 30. The last comparison
is informative: the middle canonical token arm's epoch-10 tail lead does not persist, because early
token-cosine route exceeds it at epoch 30 by `+0.009742` OOD, `+0.014602` ID, and `+0.002841` worst
experiment. Early token-cosine route also dominates early output dense at epoch 30 by
`+0.030749/+0.056658/+0.006494`. This makes early routing pressure the current leading mechanism
hypothesis, not a sparse-efficacy result.

All six remaining planned cells launched, so every one of the 43 predeclared arms has now started.
A priority refill briefly raced a still-live precomputed shard queue and created a duplicate
early-canonical dense process. It was detected after W&B startup but before epoch 0 or meaningful
training, explicitly excluded, and stopped; the queue controllers on shards 0 and 1 were stopped
without disturbing their healthy children. The kept early-canonical dense comparator is PID 74792.
The two final missing MoE arms then started directly from the exact frozen `cells()` definitions.
This yields 43 unique planned launches plus one excluded transient physical W&B start, never a
44-arm scientific screen.

Ten distinct workers now occupy all ten available H100s: container 2887 runs early token-cosine
route and late image-linear route; 2875 runs late token-linear route and exact early-canonical
dense; 2874 runs original and late token-linear canonical; 2862 runs late-canonical dense and
late-output dense; 2859 runs late token-cosine output and middle-output dense. The tested queue is
empty only because complete planned launch coverage has been reached, not because work stopped.
The exact dense comparator and epoch-60 route trajectories now shorten the critical path. OOD test
remains sealed, multiplicity and seed-0 selection remain the largest threats, and no manuscript
claim is licensed.

## 2026-08-02 05:16 EDT factorial60 adjudication and router-aux refill

The factorial60 screen is complete: all 43 predeclared seed-0 arms have preserved milestones, 34
were successively halved, and nine reached epoch 60. All nine final JSONs, exact 10/30/60 milestone
streams, logs, checkpoints, finite metrics, four-environment coverage, parameter accounting,
clean `b8ece25` provenance, `selection_split=ood_val`, `test_evaluated=false`, and null/absent
OOD-test fields pass one strict validation-only batch. Nine manifests are present. Every final
folder is published under the declared factorial60 HF prefix and its remote file list was checked.

Late canonical dense-wide is the finished mean-OOD leader at `0.214735`; early image-linear route
is the best sparse arm at `0.211082`. Relative to that best dense arm, sparse changes OOD/ID/worst
by `-0.365/-2.071/+0.406` percentage points. Relative to its same-placement early dense control it
is directionally positive at `+1.847/+3.218/+0.933` points, but misses the predeclared `+5` mean-OOD
trigger. Placement and ordinary dense adaptation are therefore the leading seed-0 explanation;
early image routing retains an exploratory tail signal, not a replicated sparse-efficacy claim.

The first three router-auxiliary finals are strictly valid and published. Zero load-balance with
router z-loss leads them at OOD `0.212401`, `+0.436` points over the factorial token-route reference
but `-0.233` below late canonical dense and with a weaker tail. Ten distinct remaining router arms
were launched across the five running 2-H100 containers, leaving three queued. The clean execution
commit is `7dcb42e` / tree `1755b2f`; its physical-GPU occupancy guard prevents controller restarts
from duplicating preserved workers. All ten running H100s were assigned at the post-refill audit.
Four pre-guard controller attempts are operational exclusions with no scientific milestone.

`tester6` 2893 was `stopped` when inspected. Exactly one authorized start request was issued this
invocation; the immediate portal recheck still reported `stopped`, so no duplicate start or new
container was requested. Next: strictly validate the router epoch-10 handoff, prune under the
predeclared rule, and immediately refill from the three queued settings. OOD test remains sealed;
seed-0 multiplicity is the largest scientific threat.

## 2026-08-02 05:45 EDT router-aux epoch-10 handoff and final queue refill

All 13 available router-auxiliary epoch-10 rows pass exact registry/run/seed/epoch checks, finite
train/ID/OOD-validation/worst-experiment metrics, four-environment coverage and count, ERM,
`selection_split=ood_val`, `test_evaluated=false`, absent/null OOD-test fields, epoch-10 checkpoint,
and fatal-scan validation. The route-pressure zero-balance plus z-loss row leads mean OOD validation
at `0.138827`; zero balance without z-loss is nearly tied at `0.138522` and has a better tail
(`0.014610` versus `0.013393`). Canonical `balance=0.01,zloss=0` has the best worst-experiment value
(`0.017045`) but lower mean OOD (`0.130505`). Thus lower balance is the sharper early mechanism,
while z-loss itself is not yet isolated. This is a seed-0, 16-setting, multiplicity-exposed
mechanism screen, not sparse efficacy.

Three strictly dominated rows were pruned only after their validated milestone and checkpoint were
preserved: route `balance=0.01,zloss=0.01`, canonical `balance=0,zloss=0.001`, and route
`balance=0.01,zloss=0.0001`. Their shard controllers immediately refilled with the final three
predeclared cells: canonical `balance=0.001,zloss=0`, canonical `balance=0.01,zloss=0.01`, and
canonical `balance=0,zloss=0`. The post-refill audit resolves ten distinct workers, two per running
2-H100 container, ten initialized W&B logs, no fatal markers, and zero idle H100s in the five
running containers. Two nonfrontier `balance=0.001,zloss=0.001` pressure-paired rows remain to epoch
30 because their matched trajectory directly tests whether the early route/canonical interaction
persists and no additional tested refill would otherwise occupy those slots.

`tester6` container 2893 was still stopped. One authorized start request was issued this invocation;
the immediate portal recheck remained stopped, so no duplicate request or replacement container was
created. Next: validate the three refill epoch-10 rows and the surviving epoch-30 trajectories,
prune/refill from the tested next-family queue when available, and require persistence without an ID
or tail loss before treating low balance as more than an early direction. OOD test remains sealed.

## 2026-08-02 06:18 EDT router-aux epoch-30 interaction handoff

Nine epoch-30 rows and 25 cumulative milestone rows across all 16 registered router-aux streams
pass exact registry/run/seed/epoch, unique ordering, finite train/ID/OOD-validation/worst metrics,
four-environment coverage and count, ERM, checkpoint, clean `7dcb42e` provenance, fatal-scan,
`selection_split=ood_val`, `test_evaluated=false`, and absent/null OOD-test checks. Six epoch-30
rows are new since the prior handoff. Three published finals and three preserved prunes remain
unchanged; no new final or HF upload is claimed.

The epoch-10 zero-balance ordering does not persist. Route pressure with `balance=0.01,zloss=0`
leads epoch-30 mean OOD at `0.198295`. Against the exact canonical-pressure arm with the same
auxiliary weights, it changes OOD/ID/worst by `+1.228/-0.207/+0.081` percentage points. Zero
balance without z-loss has the best epoch-30 tail (`0.019075`) but trails the mean leader by
`0.609` points. The sharper live explanation is therefore a route-pressure by moderate-balance
interaction with a mean-versus-tail tradeoff, not a generic zero-balance or z-loss benefit.

No worker was pruned at this handoff: the three globally dominated canonical rows are the exact
pressure-matched controls needed to separate routing pressure from auxiliary-weight effects, and
the remaining rows are Pareto-relevant or complete those declared pairs. Fresh audits resolve the
same ten direct workers, two model allocations per running container, clean execution state, 16
milestone files with 25 rows, three results, three prunes, fresh logs, and zero fatal markers.
Three refill settings have not yet emitted epoch 10. `tester6` remained stopped after the one
authorized start request in this invocation. OOD test is sealed; 16-cell seed-0 multiplicity is the
largest threat.

## 2026-08-02 06:26 EDT router-aux full epoch-10 coverage and tenth epoch-30 pair

The three latest refills emitted valid epoch-10 rows and the last pressure-paired survivor emitted
epoch 30 while the preceding handoff was being persisted. The strict campaign ledger now contains
29 rows: all 16 epoch-10 rows, ten epoch-30 rows, and three epoch-60 finals. The same exact identity,
metric, environment, checkpoint, provenance, split-blindness, and fatal-log checks pass with no
exclusion or new final.

The new `balance=0.001,zloss=0.001` pressure pair does not favor route pressure: route minus
canonical changes OOD/ID/worst by `-0.193/+0.153/-0.406` points. This makes the positive
`balance=0.01,zloss=0` pressure contrast setting-specific rather than a general route-pressure
effect. The new canonical high-z-loss row is dominated at epoch 10, but remains to epoch 30 under
the declared delayed-stabilization alternative; the other two new canonical rows are exact
pressure controls. Ten workers remain assigned and no H100 in the five running containers is idle.
OOD test remains sealed.

## 2026-08-02 07:52 EDT router-aux epoch-60 falsification and temperature refill

The strict router-aux ledger now contains 38 valid rows: all 16 epoch-10, 13 epoch-30, and nine
epoch-60 milestones. Nine final JSONs, exact milestone streams, finite metrics, four validation
environments totaling 9,854 samples, 30,676,212 parameters, ERM, clean `7dcb42e` provenance,
epoch-60 checkpoints, fatal scans, `selection_split=ood_val`, `test_evaluated=false`, and absent or
null OOD-test fields pass together. All nine completed folders now have manifests and were uploaded
under the declared router-aux HF prefix with remote file-list verification.

The predeclared persistence test is negative. At `balance=0.01,zloss=0`, route pressure exceeded
its exact canonical pair by `+1.228` OOD points at epoch 30, but only `+0.142` points at epoch 60
(`+0.401` ID and `+0.365` worst-experiment points). At `balance=0.0001,zloss=0.001`, route changes
mean OOD by `-0.335` while improving the tail by `+0.446` points. The completed mean leader remains
route zero-balance plus z-loss at OOD `0.212401`; this is a small trajectory/tail effect, not a
large stable sparse gain. The epoch-30 interaction is therefore falsified as a persistent mean-OOD
mechanism. Sixteen-setting seed-0 multiplicity remains the largest threat.

A 12-cell router-temperature-initialization question was predeclared in source commit `e22b320`:
canonical versus route pressure, two new initial temperatures (`0.03`, `0.2`), and three
representative auxiliary settings, with the existing `0.07` trajectories as shared references.
Six exact pressure-paired cells launched from the clean, previously tested `7dcb42e` execution tree
while the remaining six stay queued. Fresh process/GPU audits resolve two model allocations on each
of containers 2887, 2875, 2874, 2862, and 2859: four remaining router-aux workers plus six new
temperature workers occupy all ten available H100s. `tester6` 2893 was stopped; exactly one start
request was issued and the immediate portal state still showed stopped, so no duplicate request or
replacement was made. OOD test remains sealed.

## 2026-08-02 09:12 EDT router-aux final decision, temperature handoff, and locked-confirmation queue

All 13 router-aux survivors now have strict epoch-60 results and manifests; together with three
preserved prunes this closes all 16 registered cells at 16/13/13 valid epoch-10/30/60 coverage and
42 milestone rows. Exact run/config/seed identity, finite metrics, four OOD-validation environments
and 9,854 samples, ERM, parameter accounting, checkpoints, clean `7dcb42e` provenance, fatal scans,
`selection_split=ood_val`, `test_evaluated=false`, and absent/null OOD-test fields pass. The strict
publisher uploaded the four newly completed folders, so all 13 completed folders have manifests
under `rxrx1/cell_dino_cp5/router_aux60_20260802`.

The completed seed-0 survivor is canonical pressure with `balance=0.01,zloss=0.01`: OOD/ID/worst
are `0.212401/0.526642/0.021916`. Relative to its exact same-placement early dense comparator this
is `+1.979/+3.130/+0.649` percentage points; relative to the best late dense finalist it is
`-0.233/-2.159/+0.122`. It does not meet the `+5` mean-OOD target, but it does satisfy the separately
predeclared smaller-mean plus consistent worst-experiment pathway. Locked 60-epoch seed-1/2 sparse
and exact-dense pairs are therefore licensed with no further tuning. This remains a winner selected
from 16 router-aux cells after the prior 43-arm factorial and is not confirmatory evidence yet.

Six router-temperature epoch-10 rows pass the same strict metric, environment, checkpoint, split,
test-blindness, provenance, and fatal-scan checks. Route-minus-canonical OOD changes by `0.000`,
`-0.477`, and `+0.041` points across the three paired settings; temperature `0.03` versus `0.2`
therefore has no material early ordering. The delayed-emergence alternative remains open to epoch
30. Four completed router-aux leases refilled immediately with four distinct temperature cells, so
containers 2887, 2875, 2874, 2862, and 2859 each run two temperature workers and all ten available
H100s are assigned. Two temperature cells remain queued.

The next-family queue is no longer thin. Four locked confirmation arms, eight expert-count
architecture arms, and two remaining temperature arms passed 18 focused tests and complete dry
runs from isolated clean checkout commit/tree `4893c964` / `74e62d3b`; 14 tested arms are ready and
the ranked backlog retains 24 questions. Source commit `d592a89` declares separate W&B groups, HF
folders, fairness labels, milestones, checkpoint policy, and no OOD-test access. `tester6` 2893 was
stopped; exactly one start request was issued and not repeated. Declared W&B groups are present in
the launch commands, but no local console marker or live environment key was recoverable in this
audit, so remote W&B state is operationally unverified; persistent milestone/result artifacts and
HF manifests remain the scientific source of truth.

## 2026-08-02 09:38 EDT router-temperature epoch-30 prune and immediate refill

Ten temperature epoch-10 streams and six epoch-30 streams now pass strict registry/run/seed/epoch
identity, finite metrics, ERM, exact four-environment coverage (9,854 samples), parameter accounting,
checkpoint, clean `7dcb42e` provenance, fatal-scan, OOD-validation selection, and OOD-test-blind
checks. No temperature final result or manifest exists yet. At epoch 30, temperature `0.03` exceeds
`0.2` under `balance=0,zloss=0.001` for both pressure branches, but route pressure trails its exact
canonical pair on mean OOD at both temperatures (`-0.629` and `-0.842` points). This supports a
lower-temperature optimization-geometry direction, not sparse efficacy.

The route-pressure `temperature=0.2,balance=0,zloss=0.001` cell was strictly dominated by its exact
canonical pair on OOD, ID, and worst-experiment accuracy, and by the lower-temperature route cell on
all three axes. Its 368,514,056-byte epoch-30 checkpoint was preserved and the run was pruned. The
released container 2874 GPU0 was immediately refilled with the distinct canonical
`temperature=0.2,balance=0,zloss=0` cell (PID 47384); preflight duplicate/provenance checks passed,
live W&B run `rc731n1b` appeared in the declared group, and postlaunch memory/utilization confirmed a
healthy worker. Five containers again host two assigned H100 workers each, so the ten running H100s
have zero idle devices. `tester6` 2893 remained stopped after exactly one start request.

The temperature registry is now ten active, one queued, and one preserved prune of 12 expected;
valid milestone coverage is 10/6/0 at epochs 10/30/60. The tested ready queue retains 13 arms and the
backlog retains 24 hypotheses. The next released GPU receives the remaining route
`temperature=0.2,balance=0,zloss=0` cell unless another strict prune creates a higher-ranked paired
handoff; locked seed-1 sparse/dense confirmation pairs follow. This is a multiplicity-exposed seed-0
screen. Its sharp falsifier is loss of the temperature direction by epoch 60 or failure of the
zero-auxiliary pairs to reproduce it. OOD test remains sealed.

## 2026-08-02 10:00 EDT router-temperature epoch-30 interaction reversal

The exact `balance=0.01,zloss=0` temperature pair has now reached epoch 30, raising strict
router-temperature coverage to 10/8/0 at epochs 10/30/60. Both rows pass registry/run/seed/epoch
identity, finite metrics, ERM, four environments and 9,854 OOD-validation samples, shared exact
parameter accounting, 368,520,752/368,514,056-byte checkpoints, clean `7dcb42e` execution provenance,
fatal scans, `selection_split=ood_val`, `test_evaluated=false`, and absent OOD-test fields.

This pair reverses the earlier temperature direction: with `balance=0.01,zloss=0`, temperature
`0.2` exceeds `0.03` on mean OOD in both canonical (`+0.335` points) and route (`+0.467` points)
branches. At temperature `0.2`, route minus exact canonical changes OOD/ID/worst by
`+0.792/+0.399/-0.244` points. The mean and ID direction is potentially useful, but the tail cost
prevents a tail-safe sparse claim. Together with the opposite ordering under zero balance plus
z-loss, this rules out a global "lower initial temperature is better" explanation and instead
points to a temperature-by-auxiliary-pressure interaction. A transient optimization fluctuation or
ordinary auxiliary-loss effect remains an alternative; epoch-60 persistence and the unfinished
zero-auxiliary pair are the sharp falsifiers. No new row is strictly dominated within its exact
temperature/pressure comparison, so both remain licensed to epoch 60.

Five running containers each have two distinct owned workers (ten active H100 processes, zero idle
running H100s); no fatal marker, final result, manifest, or HF publication exists. The active process
working directories all resolve to the clean dedicated execution checkout at commit/tree
`7dcb42e` / `1755b2f`, even though the shared persistent project checkout is a dirty legacy source
workspace and is not used for execution. Seven launcher logs plus the two new run logs and the
latest refill record show initialized W&B tracking without errors. `tester6` 2893 stayed stopped
after exactly one start request and was not duplicated. Accounting remains ten active, one queued,
one pruned, with 13 ready arms and 24 backlog hypotheses. The next release launches route
`temperature=0.2,balance=0,zloss=0`; locked seed-1/2 sparse/dense confirmation follows. This remains
a multiplicity-exposed seed-0 result; OOD test is sealed.

## 2026-08-02 10:14 EDT zero-auxiliary temperature handoff

Two zero-auxiliary canonical milestones, one matched route milestone, and the first epoch-60 row
raise strict temperature coverage to 11/10/1 at epochs 10/30/60. The high-temperature canonical
refill reached epoch 10 at OOD/ID/worst `0.133956/0.307963/0.014610`; the low-temperature canonical
cell reached epoch 30 at `0.189263/0.466931/0.017045`. Its exact route pair reached epoch 30 at
`0.189872/0.460677/0.017045`, and high-temperature canonical with `zloss=0.001` reached epoch 60 at
`0.211792/0.530681/0.019886`. Across all 22 cumulative milestone rows, JSON parsing, unique
run/epoch identity, finite metrics, ERM, four environments and 9,854 OOD-validation samples,
checkpoints, clean `7dcb42e` execution provenance, fatal scans, `selection_split=ood_val`,
`test_evaluated=false`, and absent OOD-test fields pass.

At zero auxiliary weight, temperature `0.2` trails `0.03` at epoch 10 by `0.284` OOD and `0.485`
ID points while improving the worst experiment by `0.284` points. At temperature `0.03`, removing
z-loss changes epoch-30 OOD/ID/worst relative to `zloss=0.001` by `-0.548/-0.790/+0.041` points.
The now-complete low-temperature zero-auxiliary pressure pair gives route minus canonical
`+0.061/-0.625/+0.000` OOD/ID/worst points, ruling out a material sparse benefit in that exact cell.
The epoch-60 high-temperature canonical row exceeds the exact early dense comparator by
`+1.918/+3.533/+0.446` points, but is weaker on mean OOD and tail than the already locked router-aux
survivor, so it does not license another fresh-seed branch. No new row is pruned. The sharp
falsifiers remain the epoch-30 zero-auxiliary temperature comparison and full epoch-60 pairs.

Five running containers still host the same ten distinct owned workers with zero idle running
H100s; each process resolves to the clean execution checkout and all logs are fresh and fatal-free.
The remaining route `temperature=0.2,balance=0,zloss=0` arm stays next in the tested queue, followed
by the locked seed-1/2 sparse/dense confirmations and expert-count cells. `tester6` 2893 remained
stopped after exactly one fresh start request this invocation and was not duplicated. There is no
temperature final, manifest, or HF publication; W&B refill `rc731n1b` remains initialized. The
screen is exploratory and multiplicity exposed, and OOD test remains sealed.

## 2026-08-02 10:23 EDT first epoch-60 pressure pair identifies a stronger provisional survivor

The complete low-temperature `balance=0,zloss=0.001` pressure pair raises strict temperature
coverage to 11/10/3 and 24 cumulative milestones. Canonical OOD/ID/worst at epoch 60 are
`0.218693/0.527750/0.022727`; route is `0.204485/0.526962/0.016234`. Route minus canonical is
`-1.421/-0.079/-0.649` points, so within-experiment route pressure is harmful on all three axes in
this exact pair. This rules out route pressure as the source of the gain.

The canonical low-temperature row is nevertheless the strongest sparse seed-0 survivor seen so
far. Relative to exact early dense it changes OOD/ID/worst by `+2.608/+3.240/+0.731` points and
strictly exceeds the previously locked router-aux survivor by `+0.629/+0.111/+0.081`. It therefore
becomes the provisional fresh-seed lock candidate when the bounded temperature campaign closes.
The more plausible mechanism is global auxiliary regularization or router geometry, not
within-experiment specialization. Seed-0 selection across the factorial, router-aux, and
temperature screens is the largest threat; locked fresh seeds and exact dense controls are the
sharp falsifier. No OOD-test value was accessed.

## 2026-08-02 10:50 EDT five temperature finals published and all released GPUs refilled

Five temperature runs now have strict final results, manifests, and remotely verified HF folders
with seven files each. Coverage is `11/10/5` at epochs `10/30/60`. The second completed pressure
pair (`temperature=0.03,balance=0.01,zloss=0`) changes OOD/ID/worst by
`+0.162/-0.059/+0.284` route minus canonical points: a small directional mean/tail result, not a
new leader. The low-temperature canonical plus z-loss row remains the provisional seed-0 leader;
the route-pressure negative in its exact pair remains the stronger mechanism result. Multiplicity
across the preceding screens is explicit, so no seed-0 winner is confirmation.

Every released H100 received an immediate nonduplicate handoff. Container 2862/GPU1 now runs the
last temperature cell (`route,t=0.2,zero auxiliary`, PID 47170, W&B `0kye1de5`). Container 2887
runs the locked seed-1 sparse/dense pair (PIDs 32715/32716); container 2875/GPU1 and 2859/GPU1 run
the locked seed-2 sparse/dense pair (PIDs 102783/37266). The first seed-2 registry launcher failed
before model start because that clean execution checkout predates the registry script; the exact
predeclared configurations were launched manually once from the same clean tested execution tree.
All ten running H100s are assigned, while tester6 remains stopped after one start request.

Two additional bounded registries are now staged: the new temperature leader at seeds 1/2 sharing
the already-running exact dense anchors, and an active-compute-matched E4/E16 by pressure screen at
the winning low-temperature/z-loss setting sharing the completed E8 pair. Together with the eight
previous expert-count cells they restore a 14-arm candidate queue once remote tests and dry runs
pass. OOD test remains sealed.

The two new registries are now licensed rather than merely staged. A separate SciServer checkout
at execution base `7dcb42e` received byte-identical copies of the five relevant source/test files
from GitHub commit `f5ac9f4` (all SHA-256 hashes matched). Five focused tests and the complete
118-test suite passed, and dry runs expanded to exactly two leader-confirmation plus four E4/E16
pressure arms with unique run IDs and declared campaign destinations. The runnable ready queue is
therefore 14 arms, with 25 backlog hypotheses.

## 2026-08-02 11:06 EDT moderate-balance pressure does not survive as a robust effect

Strict router-temperature coverage is now `12/11/7` at epochs `10/30/60`. The completed
`temperature=0.2,balance=0.01,zloss=0` pair reaches canonical OOD/ID/worst
`0.206515/0.526421/0.020698` and route `0.206921/0.526864/0.018263`. Route minus canonical is only
`+0.041/+0.044/-0.244` points: an effective mean/ID tie with the same tail-cost direction seen at
epoch 30. The lower-temperature canonical-plus-z-loss cell therefore remains the provisional
seed-0 leader. Moderate-balance route pressure is not a reproducible robust mechanism in this
pair, further favoring global auxiliary regularization or router geometry over within-experiment
specialization. Fresh-seed sparse-versus-dense consistency remains the sharp falsifier, and the
multi-family winner search remains the largest threat.

All 30 cumulative milestone rows pass exact registry/run/seed/epoch identity, finite metrics,
ERM, four held-out validation experiments totaling 9,854 samples, checkpoint identity, clean
`7dcb42e` provenance, fatal scans, `selection_split=ood_val`, `test_evaluated=false`, and absent
OOD-test fields. The two epoch-60 workers are still finalizing and retain their GPU allocations;
all ten running H100s remain assigned, with 14 independently tested arms ready for the next
release. OOD test remains sealed.

## 2026-08-02 11:14 EDT two finals published and both releases become locked replications

The high-temperature moderate-balance canonical/route runs are now complete strict finals. Their
result, log, milestone stream, three checkpoints, and SHA-256 manifest were uploaded and remotely
re-listed at seven files per run. Router-temperature accounting is seven valid finals, four active,
one preserved prune, seven manifests, and 49 remotely verified files.

Container 2874/GPU1 immediately refilled with the locked low-temperature sparse seed-1 arm
(worker PID 49277, W&B `po7ckidl`), and container 2862/GPU0 with the corresponding seed-2 arm
(worker PID 48824, W&B `9sjnn6el`). Their exact dense seed-matched comparators were already active,
so no anchors were duplicated. The initial detached launches started correctly, but shell scope
caused only the controller-marker writes to fail; the markers were repaired once to the verified
GPU worker PIDs. Exact command, clean execution checkout, live 7.9-8.0 GiB allocation, declared
tracking group, data path, and OOD-test blindness all pass. Ten H100s are again assigned, with 12
tested nonduplicate arms still ready.

## 2026-08-02 11:27 EDT fresh-seed epoch-10 directions disagree and all pairs continue

All four locked tail-safe confirmation arms have strict epoch-10 milestones and saved checkpoints.
Seed 1 sparse minus exact dense changes OOD-validation/ID/worst-experiment accuracy by
`-0.690/+0.719/+0.933` points; seed 2 changes them by `+1.370/+2.089/-0.203` points. The mean and
tail signs therefore reverse across seeds. This does not replicate a consistent sparse advantage
at epoch 10 and makes seed-0 selection optimism plausible, but it also does not dominate either
pair on every axis. Under the locked successive-halving rule all four arms continue without tuning
to the epoch-30 falsifier.

The four rows pass exact run/seed/milestone identity, finite metrics, ERM, all four held-out
validation experiments and expected counts, checkpoint, explicit launch/train-log fatal scans,
`selection_split=ood_val`, and `test_evaluated=false`. The comparison is exact-total-parameter
matched up to the documented 378-parameter (0.001232%) implementation difference. OOD test remains
sealed and the multi-family seed-0 winner search is the largest threat.

Router-temperature coverage also rises from `12/11/7` to `12/11/8`: the low-temperature
zero-auxiliary canonical row finishes epoch 60 at OOD/ID/worst `0.206617/0.525682/0.021104`, below
the provisional low-temperature plus-z-loss leader `0.218693/0.527750/0.022727`. Its exact route
pair remains active, so the final zero-auxiliary architecture contrast stays open.

All ten H100s in the five running containers remain assigned to distinct arms. `tester6` 2893 was
stopped on inspection and remained stopped after exactly one start request; no duplicate start or
container was created. Twelve tested arms remain ready, with the E4/E16 low-temperature pressure
screen first on release and the expert-count queue behind it.

At 11:35 EDT the completed zero-auxiliary canonical temperature run passed strict final/config,
parameter, three-milestone, environment, checkpoint, clean-provenance, fatal-scan, OOD-blind, and
remote-publication checks. Its seven-file HF folder raises temperature finals/manifests/verified
files to `8/8/56`; W&B run `15uoiq6l` is recorded. Container 2875/GPU0 was immediately refilled
with the highest-ranked active-compute-matched architecture arm, low-temperature canonical E4
(controller 104305, worker 104310). The preflight found the GPU free, the run absent, the shard
dry run exact, and the five-focused/118-full-tested code-equivalent checkout unchanged. Ten GPUs
are assigned again; three E4/E16 pressure cells and eight broader expert-count cells remain ready.
W&B `vzbz70rc` is live and epoch 1 is logged. The controller's expected `.active` marker was
missing even though controller and worker were both alive; it was repaired once to worker 104310.

## 2026-08-02 11:52 EDT — capacity-starved E2 architecture bracket restores the ready queue

All ten H100s in containers 2887/2875/2874/2862/2859 remain assigned to the same ten distinct
workers, with no fatal signature and no new milestone beyond the previously validated rows. The
exact allocation is sparse/dense seed 1 on 2887, low-temperature canonical E4 plus sparse seed 2
on 2875, high-temperature zero-aux canonical plus locked leader seed 1 on 2874, locked leader seed
2 plus high-temperature zero-aux route on 2862, and low-temperature zero-aux route plus exact dense
seed 2 on 2859. Occupancy is operational state, not scientific progress.

The ready queue had fallen to eleven after three licensed arms entered service. A bounded E2 lower
bracket was therefore predeclared for both canonical and route pressure at the provisional
low-temperature setting. Its scientific question is architectural: if useful conditional
specialization causes the E8 direction, a capacity-starved E2 model should trail E4/E8 while using
both experts; an E2 tie would instead favor generic regularization over the need for a larger
expert bank. Both arms are active-compute matched only, share the completed E8 anchor, save
10/30/60 checkpoints, and remain exploratory seed 0.

GitHub registry commit `a97e9062d950fbf6171ceca744aecb94d64a5536` was transferred byte-for-
byte into a separate SciServer checkout on execution base `7dcb42e7`; the two transferred files
match SHA-256 hashes. Three focused tests and the complete 116-test base suite pass, and the dry
run enumerates six unique E2/E4/E16-by-pressure IDs with declared W&B/HF destinations. Because
canonical E4 is already active, the five remaining cells combine with eight broader expert-count
arms for thirteen tested runnable arms. The next release is licensed for canonical E16 (shard
2/6), followed by route E4, route E16, then the two E2 falsifiers.

`tester6` 2893 was stopped on inspection and received exactly one start request this invocation.
Its authoritative state is Pending: 11/16 nodes fail affinity, two are unschedulable, and three
have insufficient GPU capacity; preemption cannot help. No duplicate start or container was
created. OOD validation remains the only selection split, `test_evaluated=false`, and OOD test
remains sealed. No W&B, HF, result, or manuscript artifact changed in this update.

## 2026-08-02 12:24 EDT — fresh-seed epoch-30 effects align and released compute opens E16

All four locked tail-safe confirmation arms now have strict epoch-30 milestones and checkpoints.
Sparse minus exact-total-parameter-matched dense changes OOD-validation/ID/worst-experiment
accuracy by `+1.756/+1.886/+0.041` points at seed 1 and `+1.451/+1.500/+0.122` points at seed 2.
The two-seed means are `+1.603/+1.693/+0.081` points. This is the first aligned fresh-seed signal
on all three decision axes, but it is a modest effect below the five-point target and remains a
locked confirmation of a winner chosen after multiple seed-0 families. All four arms therefore
continue unchanged to epoch 60; no epoch-90 promotion or OOD-test evaluation is licensed yet.

The four rows pass exact run/seed/epoch identity, finite metrics, ERM, the four expected OOD-
validation experiments and counts totaling 9,854 samples, `>300 MB` checkpoints, four-log fatal
scans, `selection_split=ood_val`, `test_evaluated=false`, and absent OOD-test fields. MoE and dense
contain 30,676,212 and 30,675,834 parameters, respectively, preserving the documented 378-
parameter (0.001232%) exact-total fairness tolerance. The largest threat is selection across the
preceding seed-0 factorial, router-auxiliary, and temperature screens; the sharp falsifier is loss
of aligned mean/tail direction at the paired seed-1/2 epoch-60 checkpoint.

Router-temperature accounting is now `12/12/10` strict rows at epochs 10/30/60. Ten finals have
strict manifests and 70 remotely re-listed HF files. The completed low-temperature zero-auxiliary
route row reaches OOD/ID/worst `0.209255/0.525978/0.021104`; route minus its exact canonical pair
is only `+0.264/+0.030/+0.000` points. Route pressure is therefore setting dependent: it is
harmful with z-loss, mildly positive with zero auxiliary loss, and does not displace the canonical
low-temperature plus-z-loss seed-0 leader. The high-temperature zero-auxiliary route row is the
only active cell left in this 12-cell family.

Container 2859/GPU0 released after that final and was immediately assigned the predeclared
low-temperature canonical E16 architecture arm (controller 39104, worker 39109). Free-GPU,
no-duplicate, clean code-equivalent checkout, exact 2/6 shard dry run, persistent destination,
campaign tracking/publication, and sealed-test checks passed. The worker holds 8,046 MiB and has no
fatal signature. Its controller omitted the `.active` marker; the marker alone was repaired once
to verified worker 39109 without a restart. The active-compute parameter ledger is E2/E4/E8/E16
totals `23,584,500/25,948,404/30,676,212/40,131,828`, with active FFN-plus-router counts
`1,182,337/1,183,105/1,184,641/1,187,713`; these claims are never pooled with exact-total fairness.

Canonical E4 is strict-valid at epoch 10 with OOD/ID/worst
`0.125939/0.315178/0.015422`. Relative to the shared completed E8 anchor it changes those metrics
by `-0.852/+0.278/-0.244` points. E4 is not dominated on every axis and the expert-bank question
allows delayed emergence, so it continues to epoch 30. Ten H100s remain assigned across five
running containers, 12 tested nonduplicate arms remain ready, and tester6 remains Pending for the
unchanged scheduler reason without another start request. The next automatic refill is route E4
(shard 4/6), then route E16, canonical E2, route E2, and the broader expert-count queue.

## 2026-08-02 12:41 EDT — synchronized paper and immediate route-E4 refill

GitHub commit `9b00954f020e3fe6ce6cc06a3f5eb214f8621c86`, including the locked fresh-seed
epoch-30 validation and the explicitly exploratory manuscript paragraph, was pulled into the linked
Overleaf project. The synchronized source contains the new post-gate confirmation paragraph and
compiles to seven pages with zero errors. The only warning is the pre-existing TeX Live 2025
`\\showhyphens` compatibility warning; the local fatal-error build also passes.

The completed high-temperature zero-auxiliary canonical worker released container 2874/GPU0.
The free slot was immediately refilled with the predeclared low-temperature route-E4 architecture
arm, shard 4/6: controller 51470, worker 51475, W&B `ka08nx5w`. The dry run enumerated exactly one
pending route-E4 cell, GPU0 was physically empty, the exact result was absent, and the worker now
holds 8,014 MiB with no fatal signature. The controller again omitted its `.active` file, so the
marker alone was repaired once to the verified live worker without a restart or duplicate.

Ten H100s are assigned across the five running containers and tester6 remains Pending for the
unchanged scheduler reason. Three low-temperature architecture arms are active, three cells remain
queued, and eleven tested nonduplicate arms remain ready overall. The next release is route E16
shard 5/6, then canonical E2, route E2, and the broader expert-count queue. These comparisons remain
active-compute matched only; OOD validation is the selection split and OOD test remains sealed.

## 2026-08-02 13:22 EDT — E4 pruned, route E16 launched, and locked leader reaches epoch 30

Four new milestone rows are strict-valid. Canonical E4 at epoch 30 reaches OOD/ID/worst
`0.185508/0.466906/0.011769`, trailing its shared canonical-E8 anchor by
`-0.923/-0.793/-0.487` points. It meets the predeclared all-axis domination falsifier, so its
epoch-30 checkpoint was preserved, a `.pruned` record was written, and worker 104310/controller
104305 were stopped. This negative rules out E4 as a better low-temperature canonical bank size
for this seed and recipe; it does not rule out route pressure, E2 capacity starvation, or E16/E32.

Canonical E16 at epoch 10 reaches `0.139334/0.318625/0.015016` and differs from the shared E8
anchor by `+0.487/+0.623/-0.284` points. This Pareto tradeoff is not dominated, so E16 continues
to epoch 30. The freed container2875/GPU0 was immediately refilled by route E16: controller
106027, worker 106032, W&B `eytascl6`. The exact shard dry run, no-duplicate/result-absent/free-GPU
checks, persistent output, tracking/publication destinations, and sealed-test checks passed. A
missing `.active` marker was repaired once to the verified worker without restart.

The locked low-temperature sparse leader at seeds 1/2 is also strict-valid at epoch 30. Sparse
minus the exact dense anchors is `+0.457/+0.850/-0.081` points at seed 1 and
`+1.928/+1.576/+0.041` at seed 2, averaging `+1.192/+1.213/-0.020`. Positive mean OOD and ID
license unchanged continuation to epoch 60, but the result is below the five-point target, lacks
aligned tail improvement, and is weaker than the separately locked tail-safe epoch-30 pair. It is
not a replication claim; selection multiplicity remains the largest threat and epoch 60 is the
sharp falsifier.

The ready queue was restored from ten to twelve by a bounded canonical/route E32 extension. GitHub
commit `2e37f34819e6ddb6b4014bb17b820fe50ff3dc6c` adds the two-cell registry and test. In a separate
SciServer code-equivalent checkout on execution base/tree `7dcb42e7`/`1755b2f`, four focused and
117 full tests pass, SHA-256 identities match the local files, and two dry runs enumerate only
canonical E32 and route E32 with the declared W&B/HF destinations. Ten H100s are again assigned;
tester6 remains Pending for 11 affinity mismatches, two unschedulable nodes, and three insufficient-
GPU nodes, with preemption unhelpful and no repeat start. The next refill is canonical E2, then
route E2, canonical/route E32, and the eight broader expert-count arms. OOD test remains sealed.

## 2026-08-02 13:42 EDT — seed-1 tail-safe final validates and both GPUs refill with E2

The locked tail-safe seed-1 sparse and exact-dense pair completed epoch 60 and passes strict final
validation. Sparse reaches OOD/ID/worst `0.217780/0.536393/0.024351`; dense reaches
`0.206109/0.518591/0.022321`, for sparse-minus-dense `+1.167/+1.780/+0.203` points. The aligned
mean, ID, and tail direction persists from epoch 30, although the mean advantage contracts and
remains below +5. This is one fresh-seed final, not a completed replication: locked seed 2 is still
running and is the sharp falsifier. Both folders now have strict manifests and seven remotely
verified HF files; W&B runs are `pbuazutb` and `dv3di7mt`. OOD test was not evaluated.

The released container2887 GPUs were immediately filled with the predeclared capacity-starved E2
pair. Canonical E2 worker 37956/W&B `9q59fd1t` owns GPU0 and route E2 worker 37957/W&B
`4sxy0aps` owns GPU1. Both exact shard dry runs, result/duplicate/physical-GPU checks, persistent
destinations, and OOD-test-blind preflight passed. Both workers hold 7,956 MiB, logs are fatal-free,
and missing `.active` markers were repaired once to the live workers without restart.

Consuming the E2 pair reduced the ready queue to ten, so a bounded canonical/route E64 extreme-
overfragmentation bracket was added at GitHub commit `09a22aa086181267e4d14e5e343f60451c41a9c1`.
The SciServer code-equivalent E64 registry passed a focused disjointness/config check, the existing
117-test suite, and two exact dry runs. The two E64 cells restore twelve ready arms: eight broader
expert-count cells plus E32 and E64 pressure pairs. All ten running H100s are assigned; tester6
remains Pending for the unchanged scheduler reason. Next refill is canonical E32, route E32, the
E64 pair, then the broader queue. OOD test remains sealed.

## 2026-08-02 14:20 EDT — two-seed tail-safe confirmation closes and five GPUs refill

The locked tail-safe seed-1/2 sparse-versus-exact-dense comparison is now strict-valid at epoch 60
for all four rows. Sparse minus dense OOD-validation/ID/worst-experiment accuracy is
`+1.167/+1.780/+0.203` points for seed 1 and `+1.055/+2.593/+0.284` for seed 2, averaging
`+1.111/+2.187/+0.244`. The effect is below the five-point mean target, but its aligned positive
tail direction at both fresh seeds satisfies the predeclared smaller-mean plus consistent-tail
alternative. This is decision-grade confirmation for this exact locked recipe, not a search-free
population claim: the recipe was chosen after multiple seed-0 screens and uncertainty remains
large at two fresh seeds. A third locked seed or independent locked family losing the positive
tail direction is the sharpest falsifier. OOD test was not evaluated.

The separately locked low-temperature seed-1/2 leader also closes with mean sparse-minus-dense
`+0.761/+2.317/+0.162` points. Its aligned but weaker mean OOD result makes low temperature alone
an insufficient explanation and leaves the tail-safe auxiliary recipe as the stronger candidate.
The router-temperature family closes at 11 strict finals plus one preserved prune; the final
high-temperature zero-auxiliary route row is OOD/ID/worst `0.211082/0.525042/0.024756`.
All five new finals pass exact run/config/seed identity, finite metrics, four environments and
9,854 samples, milestones/checkpoints 10/30/60, fatal scans, clean execution provenance,
`selection_split=ood_val`, `test_evaluated=false`, and null/absent OOD-test fields. Publication is
remotely verified at 4 manifests/28 files for tail-safe, 2/14 for the leader, and 11/77 for router
temperature.

Five released H100s were immediately refilled without duplicate scientific starts: canonical E32
worker 108209 on 2875/GPU1, route E32 worker 51113 on 2862/GPU1, canonical E64 worker 41338 on
2859/GPU1, route E64 worker 53264 on 2874/GPU1, and broader expert-count route-E4 tail-safe worker
52582 on 2862/GPU0. The other live workers are canonical/route E2 37956/37957 on 2887, route E16
106032 on 2875/GPU0, route E4 51475 on 2874/GPU0, and canonical E16 39109 on 2859/GPU0. All ten
processes are owned by `idies`, hold 8.0--9.1 GB, and their recent fatal scans are clear. Latest
epochs are E2 14/14, E16 34/26, route E4 37, E32 9/9, E64 9/6, and broader route-E4 6.
Three completed stale markers were moved to explicit recoverable `stale_completed` names and five
missing live markers were repaired to verified worker PIDs without restarting workers.

Consuming five ready arms left seven, so an eight-cell E32/E64 by canonical/route by
tail-safe/no-auxiliary interaction bracket was predeclared. Source commits
`d370b701760a32ba45af2a172e5562fe587dce25` and
`3ac5ef86c804e93f809f0caa347f8b34636cfff0` add and repair the direct registry execution. In a
separate code-equivalent checkout on base/tree `4893c964e67477a25a2e3b331dde9c7e641ca669` /
`74e62d3b`, 124 full tests pass and the direct dry run enumerates exactly eight unique cells with
declared W&B/HF destinations. The ready queue is therefore 15 and the backlog is 27. tester6 2893
remains Pending: 11 affinity mismatches, two unschedulable nodes, three insufficient-GPU nodes;
preemption remains unhelpful and no repeat start/create was issued. On the next release, launch
`expert_count60` canonical-E4 tail-safe, then its zero-auxiliary comparator; after the remaining
seven broader cells, enter the extreme auxiliary bracket. OOD test remains sealed.

GitHub commit `d71aa6955dcff053c92d5f3382fe7d82cf1331ba` was then pulled into the linked
Overleaf project. The synchronized manuscript contains the locked epoch-60 two-seed paragraph and
compiles to seven pages with zero errors and the single pre-existing `\showhyphens` compatibility
warning. No scientific or operational configuration changed during the paper sync.

## 2026-08-02 14:27 EDT — E32/E64 epoch-10 checkpoints validate and all four continue

Canonical/route E32 and E64 crossed epoch 10 during the paper handoff. All four milestones pass
exact run/seed/epoch and live-command identity, finite metrics, ERM, the four expected validation
experiments and 9,854 samples, 709 MB--1.16 GB checkpoint identity, fatal scans,
`selection_split=ood_val`, `test_evaluated=false`, and sealed OOD-test checks. Relative to their
same-pressure completed E8 anchors, canonical/route E32 changes OOD/ID/worst by
`-0.091/+0.859/+0.081` and `-0.142/+1.320/+0.203` points; canonical/route E64 changes them by
`+0.386/+0.182/-0.446` and `+0.578/+1.024/-0.244`. E32 is an effective mean tie with better ID
and tail, while E64 offers a small mean/ID gain with a tail cost. No row is dominated on all three
axes, and delayed architecture specialization was predeclared, so all four continue unchanged to
epoch 30. These are active-compute comparisons only and cannot be pooled with exact-total claims.
The sharp falsifier is failure to improve mean or tail at epoch 30 together with dead or unstable
expert diagnostics; four additional seed-0 cells and the preceding screens keep the result
exploratory. No worker, ready-arm count, or OOD-test state changed.

The broader `expert_count60` route-E4 tail-safe arm also reaches a strict epoch-10 checkpoint:
OOD/ID/worst is `0.130911/0.322343/0.014205`, with a 311,739,683-byte checkpoint and a clear fatal
scan. Its same-seed, same-data-order, same-pressure route-E4 zero-auxiliary comparator has not run,
so this is a validated level only and no auxiliary-loss effect is claimed. The arm continues
unchanged to epoch 30; route-E4 zero-auxiliary is promoted to the first refill to close the exact
pair before launching other broader cells. All ten H100s remain assigned and 15 arms remain ready.
## 2026-08-02 14:47 EDT — five architecture milestones validate, route E4 prunes, exact comparator refills

Five previously unconsumed milestones pass strict run/config/seed/epoch and live-command identity,
finite metrics, ERM, four OOD-validation environments and 9,854 samples, checkpoint identity,
fatal scans, `selection_split=ood_val`, `test_evaluated=false`, and sealed-test checks. At epoch 10,
canonical E2 minus canonical E8 is `+0.122/+1.933/-0.325` OOD/ID/worst points and route E2 minus
route E8 is `-0.142/+2.452/+0.122`; both are Pareto tradeoffs and continue to epoch 30. At epoch
30, canonical E16 minus canonical E8 is `-1.390/-3.620/+0.162`, while route E16 minus route E8 is
`+0.365/-1.485/+0.081`; neither is all-axis dominated, and route E16 retains the predeclared small
mean-plus-tail signature with less than two points ID loss, so both continue to epoch 60.

Route E4 at epoch 30 is `-0.934/-0.975/-0.284` points below the same-pressure E8 anchor on all
three axes. This satisfies its predeclared pruning falsifier. Worker 51475 and its one-cell
controller exited cleanly; the 311,741,277-byte epoch-30 checkpoint and copied milestone-stream
prune record are preserved. The released container2874/GPU0 was immediately assigned to the exact
broader route-E4 zero-auxiliary comparator. Its exact one-cell dry run, result/duplicate/physical-
GPU checks, persistent destinations, active-compute fairness label, checkpoint policy, and sealed-
test checks passed. Controller 54661 started worker 54666, W&B `8an4h01j`, using the declared
group and HF prefix; the worker holds 8,014 MiB and has a clear initial fatal scan.

All ten available H100s are assigned: 2887 has canonical/route E2 (37956/37957); 2875 has route
E16/canonical E32 (106032/108209); 2874 has broader route-E4 zero-aux/route E64 (54666/53264);
2862 has broader route-E4 tail-safe/route E32 (52582/51113); and 2859 has canonical E16/E64
(39109/41338). The ready queue is 14 (six remaining broader expert-count cells plus eight extreme-
auxiliary cells), backlog 27, and tester6 2893 remains Pending for the unchanged 11-affinity,
two-unschedulable, three-insufficient-GPU reason; no duplicate start/create was issued. Local and
GitHub were clean and aligned at `a1859d8`; the clean hypothesis checkout was `cd78339`, execution
checkouts were clean/code-equivalent at `7dcb42e7` (the broader checkout at `4893c964`), while the
older SciServer source tree remains `4795202` with three pre-existing tracked training-log edits.
The result favors a route-by-expert-bank-size interaction over generic pressure, but remains
seed-0, active-compute-only, and multiplicity-exposed. Route E16 losing its mean/tail advantage at
epoch 60 or showing dead/non-reliant experts is the sharp falsifier. Next refill is broader
canonical-E4 tail-safe, then canonical-E4 zero-auxiliary. OOD test remains sealed.
## 2026-08-02 14:59 EDT — E2 epoch-30 gate validates; canonical E2 prunes and refills

The canonical/route E2 epoch-30 pair passes exact run/config/seed/epoch and live-command identity,
finite metrics, ERM, four OOD-validation environments and 9,854 samples, 283 MB checkpoint
identity, fatal scans, `selection_split=ood_val`, `test_evaluated=false`, and sealed-test checks.
Canonical E2 minus canonical E8 is `-1.421/-0.899/-0.081` OOD/ID/worst points and is strictly
dominated on every gate axis, so its checkpoint and copied milestone prune stream are preserved and
worker 37956 is stopped. Route E2 minus route E8 is `-0.071/-0.359/+0.122`, an effective mean/ID
tie with a small tail gain; it is not dominated and continues unchanged to epoch 60.

Container2887/GPU0 was immediately refilled with broader canonical-E4 tail-safe. The exact shard-4
dry run, result/active duplicate guards, physical-free-GPU check, persistent destinations,
active-compute fairness label, seed/data order, milestone checkpoint policy, and sealed-test checks
passed. Controller 41682 started worker 41687/W&B `shgndkjb`; it owns 8,010 MiB and its initial
log is fatal-free. All ten available H100s remain assigned. The broader expert-count family now has
three active and five queued cells; the tested ready queue is 13, with 27 backlog hypotheses.
tester6 2893 remains Pending for 11 affinity mismatches, two unschedulable nodes, and three
insufficient-GPU nodes; no repeat start/create was issued.

This rules out a generic capacity-starvation benefit for canonical E2 through epoch 30. The small
route-E2 tail direction remains provisional and pressure-dependent; loss of that direction at
epoch 60 or collapsed/non-reliant routing is its sharp falsifier. These are seed-0 active-compute
comparisons after many searched cells, not exact-total or confirmatory evidence. Next refill is
broader canonical-E4 zero-auxiliary, then the remaining E16 broader cells. OOD test remains sealed.
## 2026-08-02 15:20 EDT — E32 tradeoffs survive; route pressure separates E64 outcomes

Four new epoch-30 expert-count milestones pass exact run/config/seed/epoch and live-command
identity, finite metrics, ERM, four OOD-validation environments and 9,854 samples, checkpoint
identity, fatal scans, `selection_split=ood_val`, `test_evaluated=false`, and sealed-test checks.
Canonical E32 minus canonical E8 is `-0.599/+0.234/+0.162` OOD/ID/worst points, and route E32
minus route E8 is `-0.852/-1.108/+0.365`; both remain Pareto tail tradeoffs and continue unchanged
to epoch 60 under the predeclared smaller-mean tail alternative. Canonical E64 minus canonical E8
is `-0.233/-2.556/+0.000`, weakly dominated because mean and ID fall while worst experiment only
ties. Its 1,163,265,872-byte checkpoint and copied milestone prune record are preserved, worker
41338 is stopped, and its controller exits. Route E64 instead gives a small aligned
`+0.589/+0.145/+0.122` OOD/ID/worst advantage over route E8 and continues to epoch 60.

Container2859/GPU1 was immediately refilled with broader canonical-E4 zero-auxiliary. The clean
`4893c964` checkout, exact shard-5 dry run (one planned, one pending), result/active duplicate
guards, physical-free-GPU check, persistent destinations, active-compute fairness label, seed/data
order, checkpoint policy, and sealed-test checks passed. Controller 43121 started worker 43126 and
W&B `60jcriar`; it owns 8,010 MiB, is syncing in the declared group, reports 9,854 OOD-validation
samples with test untouched, and has a clear initial fatal scan. All ten available H100s remain
assigned. Temperature expert-count now has six active and four pruned rows; broader expert-count
has four active and four queued rows. The tested ready queue is 12, with 27 backlog hypotheses.
tester6 2893 remains Pending for the unchanged 11-affinity, two-unschedulable,
three-insufficient-GPU reason; no repeat start/create was issued.

The evidence rules out a monotone unconditional benefit from larger expert banks and specifically
rules out canonical E64 for this recipe through epoch 30. Pressure-specific specialization is now
the bounded remaining explanation: E32 preserves tail tradeoffs and route E64 is modestly aligned.
Loss of either direction at epoch 60 or dead/non-reliant routing is the sharp falsifier. These are
seed-0 active-compute comparisons after many searched cells, not exact-total or confirmatory evidence. Next refill is the
broader E16 tail-safe cell, then its exact zero-auxiliary comparator. OOD test remains sealed.
## 2026-08-02 15:45 EDT — Canonical E16 fails its terminal gate; canonical-E4 reaches epoch 10

Canonical E16's epoch-60 milestone passes exact run/config/seed/epoch and live-command identity,
finite metrics, ERM, four OOD-validation environments and 9,854 samples, 482,051,839-byte
checkpoint identity, fatal scan, `selection_split=ood_val`, `test_evaluated=false`, and sealed-test
checks. Relative to the exact same-pressure E8 epoch-60 anchor it is
`-1.431/-0.288/-0.203` OOD/ID/worst points. Its epoch-30 tail tradeoff
(`-1.390/-3.620/+0.162`) reverses by epoch 60 while mean OOD remains lower. This reaches the
predeclared terminal horizon with no epoch-90 or fresh-seed license. Worker 39109 is still executing
the predeclared mechanism analysis and final-result emission at high CPU activity with its model
resident; terminating it now would discard required terminal artifacts. Container2859/GPU0 will be
refilled immediately when worker 39109/controller 39104 exits.

Broader canonical-E4 tail-safe also passes a strict epoch-10 milestone: train/ID/OOD/worst are
`0.5052/0.3179/0.1351/0.0150`, with a 311,745,867-byte checkpoint. Its exact canonical-E4
zero-auxiliary comparator is active but has not reached epoch 10, so this is an unpaired level and
supports no auxiliary-loss effect claim. It continues unchanged to epoch 30. All ten H100s remain
assigned, the tested ready queue remains 12, and the backlog remains 27. tester6 2893 is still
Pending for the unchanged scheduler reason; no duplicate start/create was issued.

This rules out canonical E16 as a robust expert-bank regime at epoch 60 and shows that its earlier
tail direction was transient. The remaining bank-size explanation is pressure-specific, currently
carried by route E16/E64 and E32 tail tradeoffs. Seed-0 trajectory noise remains the alternative;
loss of those pressure-specific directions at epoch 60 or dead/non-reliant routing is the sharp
falsifier. Multiplicity is explicit and all comparisons here are active-compute, not exact-total.
Next refill is the first broader E16 tail-safe cell, with its zero-auxiliary comparator immediately
afterward. OOD test remains sealed.

## 2026-08-02 20:13 EDT — Route E16 licenses a locked fresh-seed pair and both released GPUs refill

Route-balanced E16's epoch-60 final passes strict result/config/seed identity, finite metrics, ERM,
four OOD-validation environments and 9,854 samples, all 10/30/60 milestones, a 482,044,183-byte
checkpoint, fatal scan, `selection_split=ood_val`, `test_evaluated=false`, null OOD-test fields,
and Cell-DINO/DINOv2 provenance checks. Against its exact route-balanced E8 anchor, E16 changes
OOD/ID/worst by `+0.528/-0.007/+0.528` points at epoch 60, following
`+0.365/-1.485/+0.081` at epoch 30. This is far below the five-point target but satisfies the
predeclared smaller-mean criterion: mean and worst-experiment directions are positive at both
milestones and final ID is effectively tied. Canonical E16 remains the terminal negative already
reported. Both completed E16 finals now have checksum manifests and five remotely verified HF
files each.

The licensed question is now frozen as route-balanced E16 versus E8 at fresh seeds 1 and 2, with
temperature `0.03`, zero balance loss, z-loss `0.001`, identical data order/optimizer, and the
60-epoch 10/30/60 checkpoint policy. Source commit `4969659` adds the four-cell registry. The
SciServer code-equivalent checkout is based on `4893c964`, has indexed tree
`081bd31e94145fc9cefc033bf1418417ac0f30c7`, matching source/test SHA-256, two focused tests, a
full-suite exit status of zero, and exact one-cell dry runs. Seed-1 route-E8 started on
container2859/GPU0 as worker 44982/W&B `qsbm21cg`; route-E16 started on container2875/GPU0 as
worker 110491/W&B `gviobuj4`. Both hold about 8.0 GB, report selection on 9,854 OOD-validation
samples with test untouched, and have clear initial fatal scans.

All ten available H100s are assigned: 2887 GPU0/1 workers 41687/37957; 2875 GPU0/1
110491/108209; 2874 GPU0/1 54666/53264; 2862 GPU0/1 52582/51113; and 2859 GPU0/1
44982/43126. Temperature expert-count has two completed finals, four active rows, and four pruned
rows. The confirmation family has two active and two queued rows. Four broader expert-count plus
eight extreme-auxiliary plus the seed-2 confirmation pair leave 14 exact dry-run-ready arms;
backlog remains 27. tester6 2893 remains Pending for 11 affinity mismatches, two unschedulable
nodes, and three insufficient-GPU nodes; no duplicate start/create was issued.

The result rules out a generic monotone bank-size benefit and narrows the explanation to a weak
route-pressure interaction. Only four of sixteen experts are used and route reliance is 0.00457,
so stable sparse specialization is not established. The alternative is seed-0 selection optimism
or ordinary optimization noise. Failure of the locked seed-1/2 E16-minus-E8 mean or tail direction,
or more than two ID points loss, is the sharp falsifier. This remains active-compute matched,
multiplicity-exposed, and OOD-test blind.

## 2026-08-02 20:25 EDT — Route E2 reaches its tail gate; matched E4 objective pairs remain tradeoffs

Route-balanced E2 reaches a strict epoch-60 milestone with OOD/ID/worst
`0.206312/0.526372/0.018669`. Against route E8 this is
`+0.183/-0.059/+0.244` points, after `-0.071/-0.359/+0.122` at epoch 30. The
final mean is positive, ID is effectively tied, and the worst-experiment direction is positive at
both milestones. This satisfies the predeclared smaller-mean consistent-tail clause and licenses a
locked E2 fresh-seed check sharing the already running/predeclared route-E8 seed anchors. The E2
worker remains active for mechanism analysis and final-result emission, so its GPU is not yet free.

The matched broader E4 objective comparisons also validate. At canonical epoch 10, tail-safe minus
zero-auxiliary is `+0.589/-0.404/+0.081` OOD/ID/worst points. At route epoch 30 it is
`+0.101/-1.135/-0.406`. Both are Pareto tradeoffs, so all four continue under successive halving;
however, the route tail-safe worst-environment hypothesis currently fails and must recover at
epoch 60 to remain plausible. No auxiliary objective effect is generalized across pressure.

Source commit `488e77b` adds two locked E2 rows, one per fresh seed, with exact shared E8 comparator
IDs. The remote source/test hashes match; four focused tests and the full suite pass, both exact
one-cell dry runs are pending-clean, and indexed execution tree is
`904aff13f75fec7b3b2a8b5fd7c2ac5b65fc88c4`. These two rows raise the tested ready queue to 16.
Seed-1 E2 is the exact refill for container2887/GPU1 when worker 37957 and its controller finish;
until then all ten H100s remain assigned and OOD test remains sealed.

The route-E2 seed-0 controller subsequently exited cleanly (`rc=0`) after W&B sync and final JSON
emission. Container2887/GPU1 was immediately refilled with the locked route-E2 seed-1 confirmation:
controller 43488, worker 43493, W&B run `28k2fboh`. It loaded the declared 9,854-sample OOD-validation
split, explicitly reports the test untouched, holds about 7.96 GB on GPU1, and has no fatal-log match.
The locked confirmation family therefore has three active rows and three queued rows; 15 tested arms
remain ready, all ten available H100s are assigned, and OOD test remains sealed.

## 2026-08-02 20:48 EDT — Route E32 becomes the architecture leader; four locked refills start

Canonical E32, route E32, and route E64 finish cleanly at epoch 60 with three 10/30/60 checkpoints,
complete final JSONs, W&B sync, controller `rc=0`, clear fatal scans, four OOD-validation
environments/9,854 samples, and sealed-test fields. Against their exact pressure-matched E8 anchors,
canonical E32 changes OOD/ID/worst by `-0.396/+0.305/-0.446` points, route E32 by
`+0.710/+0.025/+0.649`, and route E64 by `+0.142/+0.039/+0.609`. Canonical E32 therefore has no
promotion. Route E64 is not advanced because route E32 is stronger on mean and tail with far fewer
total parameters; it also uses only 3/64 experts with route reliance `0.00213`. Route E32 is the
seed-0 route leader and preserves its predeclared tail direction from epoch 30, so it licenses one
locked E32 seed-1/2 check. This remains exploratory: only 3/32 experts are used and route reliance is
`0.00599`.

The broader E4 sweep also moves. Canonical tail-safe minus canonical zero-auxiliary at epoch 30 is
`+1.745/+0.987/+0.203` OOD/ID/worst points, an aligned but seed-0 objective effect; both continue to
epoch 60. Route-E4 tail-safe finishes at OOD/ID/worst `0.2093/0.5236/0.0150`, but its exact zero-aux
comparator is only at epoch 43, so no route objective effect is claimed. Its randomized-route
reliance is effectively zero (`-0.00020`), which weighs against a routing-mechanism explanation.

Source commit `21d7bbe` freezes E32 seeds 1 and 2 against the already launched route-E8 anchors.
The code-equivalent SciServer tree is `fee4c45d00ea38e209a913f69ca9b0db62338b39`; source/test hashes
match, six focused tests and all 130 tests pass, and both one-cell dry runs are pending-clean. A
remote Git fetch was unavailable and changed nothing; the documented code-equivalent copy path was
used once. Four released GPUs were refilled: route-E8 seed2 worker 56371/W&B `kkkemsr0`, route-E16
seed2 112826/`abcjy5pp`, route-E2 seed2 56662/`l7vkvk81`, and route-E32 seed1 57545/`6cmjnu32`.
All report 9,854 OOD-validation samples, test untouched, expected GPU memory, and no fatal signature.

All ten available H100s are assigned: 2887 GPU0/1 workers 41687/43493; 2875 GPU0/1
110491/112826; 2874 GPU0/1 54666/56662; 2862 GPU0/1 57545/56371; and 2859 GPU0/1
44982/43126. Temperature expert-count is complete at six finals and four prunes. The expanded locked
confirmation has seven active rows and one queued E32-seed2 row. Broader expert-count has one final,
three active, and four queued; 13 tested arms remain ready and backlog remains 27. OOD test remains
sealed.

## 2026-08-02 20:56 EDT — Fresh-seed E16 tail direction weakens at epoch 10; publication repair is bounded

The first locked fresh-seed E16-versus-E8 comparison passes exact run/config/seed/epoch identity,
finite metrics, ERM, four OOD-validation environments/9,854 samples, checkpoint identity and size,
clear fatal scans, and sealed-test checks. At epoch 10, E16 minus E8 is
`+0.142/-0.492/-0.568` OOD/ID/worst points. E16 is therefore a Pareto tradeoff rather than an
all-axis loser, and both rows continue unchanged to epoch 30 under the predeclared delayed-emergence
rule. This interim fresh-seed result does not reproduce the seed-0 early tail direction and raises
seed instability as the leading alternative to a bank-size mechanism; it is diagnostic, not a
replication claim.

The checksum-manifest publication retry failed before artifact mutation because the persistent-main
`scripts.run_ccas` lacks the expected `publish_hf_run` symbol. The earlier quoting failure and this
single narrow retry created no manifest. The retry budget is exhausted for this invocation; the
license condition is a clean tested checkout with the publisher import and manifest dry path
verified before one new bounded attempt. Training is unaffected: all ten available H100s retain
their documented assignments, E32 seed2 remains the highest-ranked queued refill, and OOD test
remains sealed.

## 2026-08-02 21:02 EDT — Locked E2 is aligned at epoch 10; publication repair passes its dry gate

Locked seed-1 route-E2 reaches OOD/ID/worst `0.14552/0.34596/0.02435` at epoch 10. Against the
strictly validated shared route-E8 seed-1 anchor, the paired difference is
`+1.035/+0.150/+0.731` points. Exact run/config/seed identity, finite metrics, ERM, four
environments/9,854 samples, 283,365,884-byte checkpoint, live worker, fatal scan, OOD-validation
selection, and sealed-test checks pass. This is an aligned locked interim signal, but only one fresh
seed and one early checkpoint: E2 and E8 continue unchanged to epoch 30. Together with E16's weaker
tail at the same seed, the current bounded explanation is small-bank regularization rather than
monotone expert-bank specialization. Seed-2 or later-milestone sign loss remains the falsifier.

The four-run publication repair was moved to the code-equivalent checkout that actually contains
`publish_hf_run`. All four exact candidates pass a nonmutating strict dry validation and the encoded
worker compiles before launch. Background worker 114491 is active; its log has no failure and the
canonical-E32 checksum manifest has been created while remote verification proceeds. Completion is
not claimed until all four manifests and 28 remote files are listed. All ten available H100s remain
assigned, E32 seed2 is still the first queued refill, tester6 remains Pending for the unchanged
scheduler reason, and OOD test remains sealed.

## 2026-08-02 21:04 EDT — Four-run checksum publication is complete and remotely verified

The repaired publication worker exits successfully after strict validation, manifest generation,
upload, and remote listing. Canonical E32, route E32, route E64, and route-E4 tail-safe each have one
checksum manifest and seven verified remote files: result JSON, run log, milestone stream, three
10/30/60 checkpoints, and manifest. The three temperature expert-count runs contribute 21 files
under `rxrx1/cell_dino_cp5/temperature_expert_count60_20260802`; route-E4 tail-safe contributes
seven under `rxrx1/cell_dino_cp5/expert_count60_20260802`. The publication report is preserved at
`/home/idies/workspace/hb_publish_extreme_and_E4_2100.report.json`. This is artifact traceability,
not new efficacy evidence. All ten available H100s remain assigned and OOD test remains sealed.

## 2026-08-02 17:22 EDT — Locked seed-2 rows reject a stable early tail benefit; route-E4 tail safety fails

Four fresh epoch-10 confirmation rows pass exact run/config/seed/epoch identity, finite metrics,
ERM, four OOD-validation environments/9,854 samples, checkpoint identity, live-worker and fatal
scans, active-compute fairness, `selection_split=ood_val`, `test_evaluated=false`, and sealed-test
checks. Against the same-seed E8 anchor, seed-2 E2 is `+0.203/+0.753/-0.528` and seed-2 E16 is
`+1.319/+1.147/-0.081` OOD/ID/worst points. E32 seed 1 is
`+0.690/-0.490/-0.487` against E8 seed 1. All are Pareto tradeoffs and continue unchanged to epoch
30 under the locked delayed-emergence rule. No row is pruned or promoted. Descriptively across two
seeds at epoch 10, E2 averages `+0.619/+0.452/+0.101` and E16
`+0.731/+0.327/-0.325`; these are not population estimates.

The route-E4 zero-auxiliary row also reaches a strict epoch-60 milestone. Tail-safe minus zero-aux
is `+0.447/+0.303/-0.081` OOD/ID/worst points at epoch 60, after
`+0.101/-1.135/-0.406` at epoch 30. The predeclared route-tail protection hypothesis therefore
fails for this exact recipe: its worst-experiment difference is negative at both paired milestones,
and the completed tail-safe arm has effectively zero randomized-route reliance. The zero-aux worker
remains live only for declared mechanism analysis and final-result emission; its GPU is not free.
It receives no epoch-90 or fresh-seed license and will hand off immediately to locked E32 seed 2
after clean final artifacts.

All ten available H100s remain assigned: 2887 GPU0/1 workers 41687/43493; 2875 GPU0/1
110491/112826; 2874 GPU0/1 54666/56662; 2862 GPU0/1 57545/56371; and 2859 GPU0/1
44982/43126. Seven locked confirmation rows now have valid epoch-10 milestones, E32 seed 2 remains
the sole queued confirmation row, 13 arms remain tested-ready, and backlog remains 27. tester6
2893 is still Pending for 11 affinity mismatches, two unschedulable nodes, and three insufficient-
GPU nodes; no duplicate start/create was issued. The most plausible explanation is ordinary
capacity/optimization regularization with unstable tail effects, not useful high-cardinality sparse
specialization. Later locked milestones and route-reliance diagnostics are the sharp falsifier.
Multiplicity is explicit, active-compute and exact-total claims remain separate, and OOD test is
sealed.

The route-E4 zero-auxiliary controller then exits cleanly (`rc=0`) with a strict final JSON. Its
mechanism row uses all four experts but has route reliance only `0.00081`, reinforcing the terminal
negative tail-safety interpretation. Container2874/GPU0 immediately refills with the predeclared
locked route-E32 seed-2 arm after two focused tests, the prior 130-test suite, an exact one-cell dry
run, absent result/prune/active artifacts, duplicate-process guard, physical GPU check, declared
W&B/HF destinations, and sealed-test check pass. Controller 58235 starts worker 58240/W&B
`e7s5hj5m`; it owns about 8.3 GiB, loads 9,854 OOD-validation samples with test untouched, and its
initial fatal scan is clear. All eight confirmation rows are now active, the tested ready queue is
12, and the route-E4 zero-auxiliary seven-file checksum publication is running under a strict
one-run validator.

The first publisher payload fails syntax compilation before mutation; its single narrow retry is
compiled before launch and then completes. Route-E4 zero-auxiliary now has one checksum manifest
and seven remotely listed files: result JSON, run log, milestone stream, three checkpoints, and
manifest. The verified report is `/home/idies/workspace/hb_publish_route_E4_zero_1735.report.json`.
This closes artifact traceability only; the terminal negative, multiplicity, fairness, and sealed-
test conclusions are unchanged.

## 2026-08-02 17:56 EDT — Locked E16 aligns at epoch 30; canonical E4 auxiliary pair stays small but tail-consistent

Locked seed-1 route E2/E8/E16 epoch-30 rows pass exact run/config/seed/epoch identity, finite
metrics, ERM, four OOD-validation environments/9,854 samples, checkpoint identity, live-worker and
fatal scans, active-compute fairness, `selection_split=ood_val`, `test_evaluated=false`, and sealed-
test checks. E16 minus E8 is `+0.954/+1.157/+0.812` OOD/ID/worst points, aligned after its negative
epoch-10 tail difference; E2 minus E8 is `+0.173/-0.837/-0.244`, a Pareto tradeoff. All three
remain locked and continue unchanged to epoch 60. This is one fresh seed after a multi-arm screen,
not a replication; seed 2 or epoch-60 sign loss and negligible route reliance remain the falsifier.

The canonical E4 tail-safe and zero-auxiliary rows also reach strict epoch-60 milestones. Tail-safe
minus zero auxiliary is `+0.365/+0.219/+0.365` OOD/ID/worst points, after positive mean and worst-
experiment differences at epochs 10 and 30. This satisfies the predeclared smaller-mean,
consistent-worst clause only for a bounded auxiliary-regularization confirmation. It does not show
a sparse or routing-specific advantage: locked fresh seeds and exact dense objective controls are
required. Both workers remain live for declared mechanism/final artifact emission, so no GPU is
free and no refill is launched.

All ten available H100s remain assigned: container2887 GPU0/1 workers41687/43493; 2875 GPU0/1
110491/112826; 2874 GPU0/1 58240/56662; 2862 GPU0/1 57545/56371; and 2859 GPU0/1
44982/43126. The tested ready queue remains 12 (four broader E16 cells plus eight extreme-
auxiliary cells), backlog remains 27, and tester6 container2893 remains Pending for the exact
11-affinity/two-unschedulable/three-insufficient-GPU reason without a duplicate start or create.
OOD test remains sealed.

## 2026-08-02 18:08 EDT — Canonical E4 finals close; broader route-E16 pair refills both releases

Both canonical-E4 controllers exit cleanly with strict final JSONs, W&B sync, exact Cell-DINO and
DINOv2 provenance, null OOD-test fields, and four-expert mechanism rows. Tail-safe and zero-
auxiliary randomized-route reliance are only `0.00213` and `0.00041`, sharpening generic auxiliary
regularization as the alternative to useful routing specialization. A strict two-run checksum
publisher is active for two manifests and 14 expected remote files; completion is not yet claimed.

Container2887/GPU0 immediately starts broader route-E16 tail-safe (controller45598, worker45603,
W&B `9ycfe34j`) and container2859/GPU1 starts its exact route-E16 zero-auxiliary comparator
(controller47666, worker47671, W&B `pujjny7w`). Exact shard-2/3 dry runs each report one planned,
one pending cell; result/prune/active and duplicate guards, physical GPU checks, persistent
destinations, matched seed/data/checkpoint policy, and sealed-test checks pass. Both startup logs
load 9,854 OOD-validation samples with test untouched and have no fatal match. Their missing active
markers were repaired once to the verified worker PIDs without restart. Ten H100s are again
assigned. Ten tested arms remain ready; the licensed canonical-E4 fresh-seed registry is being
prepared to restore the minimum twelve before the next release.
## 2026-08-02 18:22 EDT — Canonical publication closes; locked auxiliary queue reaches fourteen

The canonical-E4 pair now has two checksum manifests and all 14 expected remote files verified
under `rxrx1/cell_dino_cp5/expert_count60_20260802`. Publication changes traceability only. A
separate clean checkout predeclares the four locked canonical-E4 tail-safe/zero-auxiliary seed-1/2
confirmation cells at source commit `d5d2b63`. Its exact script SHA-256 is
`548790a06fa7ec422c4d9ec9d42962c15f3a994154f60ec3dc085e3459fe160a`; checksum-semantic checks,
four exact one-cell dry runs, and the existing full suite exit zero. Together with two remaining
broader E16 cells and eight extreme-auxiliary cells, the tested ready queue is 14 and the backlog is
27. A private GitHub fetch on SciServer failed before mutation because no credential was available;
the exact source payload was instead checksum-verified in memory. Launch therefore requires that
same exact source transport or a clean credentialed checkout. OOD test remains sealed.

## 2026-08-02 18:29 EDT — Seed 2 reverses E16 at epoch 30; E32 remains mixed and exploratory

Five newly available locked milestones pass exact identity, finite metric, ERM, four-environment/
9,854-sample, checkpoint, fatal-scan, OOD-validation-selection and sealed-test checks. At epoch 30,
seed-2 E16 minus E8 is `-0.690/-0.192/-0.365` OOD/ID/worst points, reversing seed 1's aligned
advantage; the descriptive two-seed mean is only `+0.132/+0.483/+0.223`. Seed-2 E2 is dominated by
E8 at `-1.817/-1.684/-0.284`. E32 seed 1 is aligned over E8 at epoch 30
(`+0.903/+1.088/+0.731`), while E32 seed 2 at epoch 10 improves mean and ID but loses tail
(`+1.187/+1.189/-0.487`). No row is pruned because the predeclared epoch-60 adjudication is needed
to resolve this seed and horizon heterogeneity. All findings remain exploratory and multiplicity-
exposed; useful specialization is falsified if epoch 60 lacks a consistent mean-or-tail advantage
with acceptable ID retention and non-negligible routing reliance.

## 2026-08-02 18:50 EDT — Wider route-E16 shows no early tail-safe benefit

The exact broader route-E16 tail-safe/zero-auxiliary pair reaches strict epoch-10 milestones. Both
rows pass exact run/config/seed/epoch identity, finite ERM metrics, the expected four OOD-validation
environments and 9,854 samples, 482 MB checkpoint identity, live-worker and fatal scans, declared
W&B identity, `selection_split=ood_val`, `test_evaluated=false`, and sealed-test checks. Tail-safe
minus zero auxiliary is `-0.274/+0.190/+0.000` OOD/ID/worst points. This rules out an immediate
mean-or-tail benefit for the auxiliary objective in the wider route-balanced E16 recipe; it does
not yet rule out the predeclared delayed effect. Both rows continue unchanged to epoch 30 without
promotion.

The leading explanation is now generic or trajectory-sensitive regularization rather than a robust
tail-protection mechanism. That explanation is falsified if tail-safe fails to recover a
nonnegative mean-and-tail direction by epoch 30 or 60; any routing-specific explanation also
requires non-negligible final randomized-route reliance. The contrast is seed 0 after extensive
architecture and objective searches, so it remains exploratory and multiplicity-exposed.

All ten available H100s remain assigned to distinct live workers: container2887 GPU0/1
45603/43493; 2875 GPU0/1 110491/112826; 2874 GPU0/1 58240/56662; 2862 GPU0/1
57545/56371; and 2859 GPU0/1 44982/47671. The locked expert-count confirmation has 8/8 active,
8/8 epoch-10 and 7/8 epoch-30 coverage; broader expert-count has four finals, two active route-E16
rows and two queued canonical-E16 rows. Fourteen tested arms remain ready and backlog remains 27.
tester6 container2893 is still Pending for 11 affinity mismatches, two unschedulable nodes, and
three insufficient-GPU nodes; no duplicate start or create was issued. OOD test remains sealed.

Locked route-E32 seed 2 subsequently reaches its strict epoch-30 milestone. Against the exact
same-seed E8 anchor, E32 is `-0.832/+0.032/+0.528` OOD/ID/worst points. Seed 1 was
`+0.903/+1.088/+0.731` at the same milestone, so the worst-experiment direction is positive in
both fresh seeds while the mean direction disagrees; the descriptive two-seed mean is
`+0.036/+0.560/+0.629`. This is a promising locked tail signal but not an overall replication,
and the active-compute estimand changes total parameter count. Both rows continue unchanged to
epoch 60. The falsifier is loss of the positive tail direction in either seed at epoch 60, more
than two ID points lost, or negligible route reliance with the effect explained by exact controls.

Route-E16 seed 1 also emits a strict epoch-60 milestone at OOD/ID/worst
`0.21656/0.53846/0.02557` with its 482,054,774-byte checkpoint. Its exact route-E8 seed-1 anchor
has not yet reached epoch 60, so this is recorded as an unpaired level and no E16-minus-E8 effect,
promotion, or replication claim is made. Worker110491 remains live for the declared final mechanism
and result artifacts; its GPU is therefore not released. The paired final contrast and route-
reliance diagnostics are the only license for an E16 conclusion.

## 2026-08-02 19:24 EDT — E16 seed 1 closes and publishes; two more locked epoch-60 levels remain unpaired

Locked route-E16 seed 1 exits cleanly and its strict final preserves the epoch-60 OOD/ID/worst
level `0.21656/0.53846/0.02557`. Randomized-route reliance is `0.01147`, larger than the nearly
zero canonical E4 mechanism values but still not a bank-size effect: the exact route-E8 seed-1
epoch-60 anchor is absent. The final JSON, run log, milestone stream, three checkpoints, checksum
manifest, and all seven remote paths pass strict validation and listing checks. The recorded
code-equivalent provenance remains execution commit/tree `4893c964` /
`fee4c45d00ea38e209a913f69ca9b0db62338b39`; `git_dirty=true` is preserved rather than relabeled.

Route-E2 seed 1 and route-E8 seed 2 independently reach strict epoch-60 milestones at
`0.21250/0.53157/0.02029` and `0.20956/0.53346/0.02273` OOD/ID/worst. Both have finite ERM metrics,
four environments/9,854 samples, exact 283,365,884/368,517,372-byte checkpoints, clear fatal scans,
OOD-validation selection, and sealed test. They are different seed pairings and remain live for
final mechanism/result emission, so no subtraction, promotion, or replication claim is made.
Locked coverage is now 8/8 at epochs 10 and 30 and 3/8 at epoch 60.

Container2875/GPU0 immediately refills with the predeclared canonical-E16 tail-safe arm
(launcher116479, controller116485, worker116490, W&B `lbxzw1ou`). Four focused tests, the 124-test suite, exact
shard-6/8 one-cell dry run, result/prune/active/duplicate and physical-GPU guards, persistent
tracking/checkpoint destinations, and sealed-test checks pass. Its missing active marker is repaired
once to the verified worker PID without restart. Ten distinct H100 workers are assigned and zero
running H100s are idle. The tested ready queue is 13 (one broader canonical-E16 cell, eight extreme-
auxiliary cells, four locked auxiliary-confirmation cells); backlog remains 27. tester6 container2893
remains Pending for 11 affinity mismatches, two unschedulable nodes, and three insufficient-GPU
nodes without a duplicate start or create.

During the closing audit route-E2 seed 1 also exits `rc=0`; its route reliance is only `0.00223`.
Its result/log/milestones/three checkpoints/manifest are strict-published as seven verified remote
files, while the missing route-E8 seed-1 epoch-60 anchor still forbids an E2 effect. The released
container2887/GPU1 immediately starts canonical-E16 zero auxiliary (controller48228, worker48233,
W&B `f7unxpd2`) after the same 4-focused/124-full test evidence, an exact shard-7/8 one-cell dry
run, and all launch guards. Its missing marker is repaired once to worker48233 without restart.
The broader E16 pair is now fully active, all ten running H100s are assigned, and the ready queue
is exactly 12: eight extreme-auxiliary plus four locked auxiliary-confirmation cells.

## 2026-08-02 19:49 EDT — Seed-2 E2 is a mean–tail tradeoff; extreme E32 tail-safe refills

Route-E2 seed 2 exits `rc=0` and strict-publishes one manifest/seven files. Against its exact
route-E8 seed-2 epoch-60 anchor, E2 minus E8 is `+0.710/-0.064/-0.284` OOD/ID/worst points. The
mean improves with negligible ID loss, but worst-experiment accuracy declines; E2 route reliance
is effectively zero at `-0.00061`. This is a locked one-seed Pareto tradeoff, not a robust sparse
win. Seed-1 pairing and the E8 seed-2 final mechanism row remain required.

Container2874/GPU1 immediately starts the highest-ranked extreme-auxiliary arm,
canonical-E32 tail-safe at temperature 0.03 (controller60303, worker60308, W&B `pbtc6tm0`). Its
exact shard-0/8 dry run, result/prune/active/duplicate and physical-GPU guards, persistent tracking,
checkpoint, and sealed-test checks pass; a missing marker is repaired once without restart. The
startup log loads 9,854 OOD-validation samples with test untouched and has no fatal match.

Because that refill would leave only 11 ready arms, a separate clean checkout opens a bounded
four-cell smoother-routing addendum: route E32/E64 at temperature 0.1 crossed with tail-safe and
zero auxiliary. The exact source is commit `345ff7b`; five focused tests, the 127-test suite, four
disjoint one-cell dry runs, checksum-verified code-equivalent transport, and a complete registry
pass. The queue is restored to 15 (seven remaining temperature-0.03 extreme cells, four smoother-
routing cells, four locked auxiliary-confirmation cells), backlog 27, and ten H100s assigned.

## 2026-08-02 20:08 EDT — Seven locked finals close the E2/E16 question; four GPUs refill

Route-E8 seeds 1/2, route-E16 seed 2, and route-E32 seed 1 are now strict finals. Together with the
three prior finals this gives 8/8 epoch-10, 8/8 epoch-30, and 7/8 epoch-60/final coverage; only
route-E32 seed 2 remains active. Exact same-seed effects versus route-E8 are E2 seed1/2
`+0.162/-0.470/-0.162` and `+0.710/-0.064/-0.284`, E16 seed1/2
`+0.568/+0.219/+0.365` and `+0.660/+0.094/-0.325`, and E32 seed1
`+0.507/-0.101/+0.122` OOD/ID/worst points. E2 and E16 are therefore below the +5-point material
target in both fresh seeds; E2 loses tail twice, and E16's tail direction flips. Route reliance is
small (`-0.00061` to `0.01147`). Neither bank size advances to epoch 90 or more seeds. The leading
alternative is generic bank regularization/optimization noise rather than reusable routing.

The four releases are refilled immediately with temperature-0.03 extreme arms: container2862
GPU0 canonical-E32 no-aux controller60195/worker60206, container2862 GPU1 canonical-E64 tail-safe
controller60196/worker60205, container2859 GPU0 canonical-E64 no-aux controller51238, and
container2875 GPU1 route-E32 tail-safe controller117930. Exact one-cell dry runs passed before
launch; the first two workers were verified at 8,394/8,848 MiB. The other six previously assigned
workers remain live at their audits, so all ten running H100s are assigned and none is idle.

Four launches reduced ready capacity to 11. Commit `71eb1f1` expands the bounded temperature-0.1
screen without changing the original route-cell shard identities: canonical/route pressure x
E32/E64 x tail-safe/no-auxiliary is now eight cells. The pressure cross separates smoother routing
from within-environment balancing. This restores the declared queue to 15 (three remaining
temperature-0.03 extreme, eight smoother-temperature, four locked auxiliary-confirmation) with
backlog 27. Its sharp falsifier remains route reliance above 0.01 plus nonnegative matched mean and
tail without more than two ID points lost. OOD test remains sealed.
## 2026-08-02 20:43 EDT — Locked E32 closes small-positive; route-E16 tail objective misses mean

The final locked route-E32 seed-2 comparison is strict: E32 minus same-seed E8 is
`+1.238/+0.660/+0.081` OOD-validation/ID/worst-experiment points. Seed 1 was
`+0.507/-0.101/+0.122`. Both fresh seeds point positive on mean and tail, but the effect is far
below the predeclared +5-point material target and route reliance remains only `0.0047--0.0063`.
The active-compute bank-size family is now 8/8 final. It rules out a large E32 routing-capacity
effect for this locked recipe; no epoch 90 or additional seeds are licensed.

The broader route-E16 tail-safe pair reaches epoch 60. Tail-safe minus zero auxiliary is
`-0.223/-0.049/+0.081` OOD/ID/worst points: a tiny tail-only tradeoff, not a mean robustness gain.
Both workers remain live only for declared final mechanism/result emission. Their next release is
not licensed for epoch 90 or fresh seeds.

Container2874/GPU0 immediately refills with temperature-0.03 route-E32 zero auxiliary
(controller62810, worker62815, W&B `l06751x1`) after 7 focused and 127 full tests, exact shard-5/8
dry run, duplicate/result/marker/physical-GPU guards, persistent destinations and sealed-test
checks. Five running 2H100 containers therefore retain ten distinct assigned workers and zero
idle running H100s; tester6/2893 remains scheduler-pending for 11 selector mismatches, two
unschedulable nodes and three insufficient-GPU nodes. Ready is 14 and backlog 27.

A five-run locked publication bundle passed strict local artifact validation and completed with
five checksum manifests and 35 remotely listed files under the existing confirmation campaign.
The initially stale smoother-screen checkout was repaired to the exact committed script, test,
and eight-cell registry hashes; seven focused tests, the full suite (exit zero), and eight disjoint
one-cell dry runs pass. OOD test remains sealed.

The route-E16 final-emission workers then released container2887/GPU0 and container2859/GPU1.
Both slots immediately refill with the remaining temperature-0.03 route-E64 pair: tail-safe
controller51136/worker51141/W&B `azo0b8ss`, and zero-aux controller52606/worker52611/W&B
`rs6sqaxa`. Exact shard-6/8 and shard-7/8 dry runs each report one unique pending cell; startup
memory is 8,842/8,844 MiB, W&B sync is active, the 9,854-sample OOD-validation path is loaded,
test is untouched, and fatal scans are clear. All eight low-temperature cells are now active;
the tested ready queue remains 12 (eight smoother cells and four locked auxiliary confirmations).

## 2026-08-02 21:34 EDT — Canonical E32 tail-safe is pruned; smoother routing refills; queue restored

Two exact active-compute pairs reach strict epoch-30 milestones. For canonical E16, tail-safe minus
zero auxiliary is `-0.507/+3.238/+0.446` OOD-validation/ID/worst points. This is a Pareto
tradeoff, not a mean robustness gain; both rows continue unchanged to epoch 60 because improved ID
and tail remain inside the predeclared objective. For temperature-0.03 canonical E32, tail-safe is
dominated at both epochs 10 and 30, with the epoch-30 difference
`-0.690/-0.982/-0.203`. The delayed-emergence exception is exhausted, so only that tail-safe row is
pruned after preserving its valid epoch-30 checkpoint and a typed prune record.

Container2874/GPU1 immediately refills with the highest-ranked smoother-routing row, route-E32
temperature 0.1 tail-safe (controller64174, worker64179, W&B `0ibgay6f`). Exact identity and
duplicate/result/marker/physical-GPU guards, persistent destinations, sealed-test checks, and the
one-cell dry run pass; the worker is verified at epoch 5 and 8,372 MiB with a clear fatal scan.
All ten available H100s again have distinct assignments and none of the five running containers is
idle. tester6/2893 could not be directly re-read because the new portal session requires login; no
credential was entered and no duplicate start/create was issued. Its last exact verified state
remains scheduler Pending for 11 affinity mismatches, two unschedulable nodes, and three
insufficient-GPU nodes.

Commit `a77b57b` adds both strict paired validations and a terminal four-cell route-pressure
temperature-0.3 addendum across E32/E64 and tail-safe/zero auxiliary. In its isolated execution-base
checkout, three focused tests, all 127 tests, and four exact one-cell dry runs pass after one
collection-only import repair. Ready is therefore 15 (seven remaining temperature-0.1 cells, four
locked E4 auxiliary confirmations, four temperature-0.3 cells); backlog remains 27. The route-E16
pair passed strict publication validation and its two-run/14-file upload is active but is not counted
complete until both manifests and the remote report exist. OOD test remains sealed.

## 2026-08-02 21:55 EDT — Extreme-bank early wave is mixed; canonical E64 reverses at epoch 30

Three exact temperature-0.03 active-compute pairs pass strict epoch-10 validation. Tail-safe minus
zero auxiliary is `+0.213/-1.248/+0.000` OOD-validation/ID/worst-experiment points for route E32,
`-1.106/-2.226/-0.041` for route E64, and `-1.624/-1.674/-0.203` for canonical E64. The route-E64
row is dominated at this milestone, but its final routing-reliance signature is unavailable and
delayed emergence was predeclared, so no unsupported prune is made. Route E32 is a Pareto tradeoff.

Canonical E64 then reverses at epoch 30: tail-safe minus zero auxiliary becomes
`+0.142/+1.214/+0.244` points. The aligned effect is small and seed-0 exploratory, so both rows
continue to epoch 60 without promotion. This trajectory weakens a simple always-harmful auxiliary
story but does not establish sparse robustness; generic regularization or optimization noise
remains the leading alternative. The sharp falsifier is loss of mean/tail direction at epoch 60 or
route reliance at or below `0.01`.

The smoother route-E32 temperature-0.1 tail-safe row reaches a strict absolute epoch-10 level of
`0.12503/0.30011/0.01664` OOD/ID/worst. Its exact zero-auxiliary comparator is tested and queued but
not yet launched, so no effect, prune, or promotion is claimed. The next physical release launches
that comparator. The route-E16 publication report now verifies two manifests and fourteen remote
files. All ten running H100s still have distinct live workers; ready remains 15 and backlog 27.
tester6/2893 received no duplicate mutation; its last exact state remains scheduler Pending for
eleven selector mismatches, two unschedulable nodes, and three insufficient-GPU nodes. OOD test is
sealed and all new rows remain multiplicity-exposed seed-0 exploration.

## 2026-08-02 22:05 EDT — Canonical E16 releases; exact smoother comparator refills

Canonical-E16 tail-safe completes its final mechanism row at
`0.20875/0.52285/0.02151` OOD-validation/ID/worst-experiment accuracy. All sixteen experts are used,
but routing entropy is `0.99995` and randomized-route reliance is `-0.00051`; this makes conditional
routing an implausible explanation for the absolute result. Its exact zero-auxiliary comparator is
still active and not final, so the auxiliary effect remains unidentified and no epoch-90 or fresh-
seed continuation is licensed.

The released container2875/GPU0 immediately starts the exact smoother route-E32 temperature-0.1
zero-auxiliary comparator (controller119744, worker119749, W&B `v8b72q7p`). Exact shard 1/8 exposes
one unique pending cell, the physical GPU is free, persistent result/checkpoint/W&B/HF destinations
are declared, and startup loads 9,854 OOD-validation samples with test untouched. The worker reaches
8,304 MiB with a clear fatal scan. A missing active marker is repaired once to the verified worker
PID without restart. Ten distinct workers again occupy the ten running H100s; ready is 14 and
backlog 27. The next gate is the exact smoother pair at epoch 10, while the next release receives
the highest-ranked remaining temperature-0.1 cell.

## 2026-08-02 22:23 EDT — Canonical E16 closes tiny-aligned; tester6 recreated; GPU refilled

The exact canonical-E16 epoch-60 objective pair is now strict-valid. Tail-safe minus zero auxiliary
is `+0.183/+0.510/+0.081` OOD-validation/ID/worst-experiment points, reversing the epoch-30 mean
deficit but remaining far below the material `+5`-point gate. Both final routers are non-reliant:
tail-safe has entropy `0.99995` and route reliance `-0.00051`, while zero auxiliary has entropy
`0.95195` and reliance `0.00274`. Both controllers exited rc0 with fatal-clear logs. This screen is
terminal-negative for a material routing-mediated effect; no epoch-90 or fresh-seed continuation is
licensed.

The old tester6 record (container2893) was absent from the authenticated Compute table. Under the
standing absent-container rule, exactly one replacement named tester6 was created as container2899
and exactly one start request was issued. It is scheduler Pending, not Running: 11 nodes fail the
affinity/selector, two are unschedulable, and three lack GPU capacity; preemption is not helpful and
has no victim. No duplicate exists and no second create/start is permitted.

The zero-auxiliary release on container2887/GPU1 immediately starts route-E64 temperature-0.1
tail-safe after exact shard 2/8 reports one pending unique row and the result/prune/active,
physical-GPU, persistent-destination, checkpoint, tracking, and sealed-test guards pass.
Controller53230 launches worker53250/W&B `qpo0u4ao`; the model holds 8,844 MiB and loads 9,854
OOD-validation examples with test untouched. All five running 2H100 containers again expose ten
distinct live GPU workers, so zero running H100s are unassigned. Ready is 13 and backlog is 27. The
next release receives the paired route-E64 temperature-0.1 zero-auxiliary arm. OOD test remains
sealed and the seed-0 contrast is explicitly multiplicity-exposed.

The canonical-E16 publication attempt passed local strict validation and created the tail-safe
checksum manifest, then the remote service rejected the first checkpoint with HTTP 400 because the
private repository storage limit is reached. The remote folder therefore contains only the
tail-safe result, log, and milestone stream (3 of 14 expected pair files); zero auxiliary has no
remote files and no local manifest yet. No completion is claimed and no blind retry is allowed. The
retry license is restored only when private storage quota becomes available.

## 2026-08-02 23:01 EDT — Three strict finals, a third refill, and canonical temperature controls

Temperature-0.03 route E32 tail-safe reverses from an epoch-10 mean-only tradeoff to three-axis
domination at epoch 30: tail-safe minus zero auxiliary is `-0.274/-0.611/-0.041`
OOD-validation/ID/worst points. Its already-near-final worker is preserved through declared final
emission and exits rc0 at `0.20733/0.51610/0.01786`; routing entropy is `0.99998` and randomized-
route reliance is only `0.00183`. This rules out delayed routing specialization for that exact
tail-safe row, but its zero-auxiliary final is still active, so the epoch-60 objective effect remains
withheld. Route E64 instead reverses by epoch 30 to a small `+0.568/+0.628/-0.041` mean/ID/tail
tradeoff and continues to epoch 60 without promotion. The smoother route-E32 temperature-0.1 pair
is dominated at epoch 10 by `-0.294/-1.539/-0.041`, but continues only through its predeclared
epoch-30 delayed-mechanism adjudication.

Two additional temperature-0.03 rows close strict-valid: canonical-E32 zero auxiliary at
`0.20773/0.52539/0.01542`, and canonical-E64 tail-safe at `0.20367/0.51576/0.01218`. Both controller
logs end rc0, fatal scans are clear, milestone/checkpoint identities pass, all four environments
and 9,854 OOD-validation examples are present, and every held-out field remains null. Canonical E32
has no epoch-60 pair because its tail-safe row was validly pruned at epoch 30; canonical E64 remains
unpaired until zero auxiliary finishes. None licenses epoch 90 or fresh seeds.

The two released container2862 GPUs immediately refill with route-E64 temperature-0.1 zero
auxiliary (controller63477/worker63497/W&B `lwy5zr93`) and canonical-E32 temperature-0.1 tail-safe
(controller63561/worker63582/W&B `c8obo0tz`). A later route-E32 tail-safe release on
container2875/GPU1 immediately refills with canonical-E32 temperature-0.1 zero auxiliary
(controller121334/worker121339/W&B `21dzaiwc`). Exact one-cell shards, free physical devices,
duplicate/result/prune/active guards, persistent destinations, OOD-validation loading, sealed-test
checks, and fatal scans pass. All ten running H100s again carry distinct workers; tester6/2899
remains scheduler Pending for the exact recorded affinity, unschedulable-node, and GPU-capacity
reason.

Commit `751dbd3` predeclares four canonical-pressure temperature-0.3 E32/E64 x auxiliary controls,
so any route-pressure temperature-0.3 movement can be separated from generic smoothing. In the
isolated execution-base checkout, six focused tests, all 130 tests, and four exact one-cell dry runs
pass. After three refills, ready is 14: two remaining temperature-0.1 cells, four locked E4 fresh-
seed cells, four route-pressure temperature-0.3 cells, and four canonical-pressure temperature-0.3
controls. Backlog remains 27; OOD test is sealed and multiplicity remains explicit.

## 2026-08-02 23:18 EDT — Canonical smoother E32 reaches epoch 10; early pressure contrast is flat

Canonical-pressure E32 temperature 0.1 tail-safe reaches a strict epoch-10 milestone at
`0.12949/0.30154/0.01583` OOD-validation/ID/worst-experiment accuracy. Against the already valid
route-pressure tail-safe row at the same E32, temperature, seed, parameter counts, data order, and
horizon, route minus canonical is `-0.447/-0.143/+0.081` points. This secondary pressure contrast
has no early mean or ID advantage and only a tiny tail tradeoff; it is not the registry's exact
auxiliary comparator, so no auxiliary-effect claim or prune is made. Both rows continue unchanged
to epoch 30, and the exact canonical zero-auxiliary epoch-10 row remains pending.

Both milestone JSONs are finite and parseable, cover four environments and 9,854 OOD-validation
examples, select on `ood_val`, declare `test_evaluated=false`, have nonempty epoch-10 checkpoints,
match the predeclared E32 parameter counts, and have live W&B sync and fatal-clear logs explicitly
stating that test is untouched. All five running 2H100 containers retain ten distinct worker PIDs;
zero allocated H100s are unassigned. tester6/2899 remains scheduler Pending for the unchanged
11-affinity, two-unschedulable, three-insufficient-GPU reason. Ready remains 14 and backlog 27.

The result is exploratory seed 0 after multiple pressure, temperature, bank-size, objective, and
architecture screens. The leading alternative remains generic optimization or auxiliary
regularization. The sharp falsifier is aligned exact auxiliary-pair movement at epoch 30 plus final
route reliance above `0.01` without more than two ID points lost.

## 2026-08-02 23:31 EDT — Smoother E32 reverses positively at epoch 30; extreme E32 closes negative

Two exact active-compute route-E32 auxiliary pairs pass strict milestone validation. At temperature
0.03, tail-safe minus zero auxiliary closes epoch 60 at `-0.233/-0.623/+0.041`
OOD-validation/ID/worst-experiment points. Across epochs 10/30/60 its mean effect is
`+0.213/-0.274/-0.233`, while the tail effect is `0.000/-0.041/+0.041`; the already validated
tail-safe route reliance is only `0.00183`. This is terminal-negative for a material tail-safe
effect in that exact recipe. No epoch 90 or fresh seed is licensed; the live zero-auxiliary worker
is retained only for declared final mechanism/result emission before immediate refill.

At temperature 0.1, the exact route-E32 pair instead reverses from
`-0.294/-1.539/-0.041` at epoch 10 to `+1.208/+0.731/+0.365` points at epoch 30. The aligned
mean, ID, and tail direction survives the delayed-emergence gate and licenses both rows to continue
unchanged to epoch 60. It remains below the +5-point material target and does not license epoch 90
or fresh seeds. Smoother optimization is now more plausible than the extreme-temperature recipe,
but generic auxiliary regularization or trajectory noise remains an alternative until final route
reliance is available.

All four milestone rows match exact run/config/seed/epoch and registry parameter identities, contain
finite ERM metrics, cover four environments and 9,854 OOD-validation examples, have nonempty
checkpoints, active W&B streams, fatal-clear logs, `selection_split=ood_val`, and
`test_evaluated=false` with explicit test-untouched logging. Ten distinct workers still occupy the
ten allocated H100s; tester6/2899 remains Pending for 11 affinity mismatches, two unschedulable
nodes, and three insufficient-GPU nodes. The exact shard-6/8 dry run for the next refill,
canonical-E64 temperature-0.1 tail-safe, reports one unique pending cell. Ready remains 14 and
backlog 27.

The smoother canonical-E32 zero-auxiliary comparator then reaches epoch 10. Tail-safe minus exact
zero auxiliary is `-0.254/-1.347/-0.487` OOD/ID/worst points, so the auxiliary row is dominated on
all three currently observed axes. Routing reliance is not available at this horizon and delayed
emergence was explicitly predeclared; both rows therefore continue only to epoch 30, without
promotion. Together with route-E32's negative epoch-10 contrast and positive epoch-30 reversal,
this makes a delayed optimization effect more plausible than an immediate benefit, while leaving
generic regularization and trajectory noise unresolved.

The extreme route-E32 zero-auxiliary worker then emits a strict final and exits rc0. It uses all 32
experts, but routing entropy is `0.92833` and randomized-route reliance is only `0.00223`; the
tail-safe comparator is likewise non-reliant at `0.00183`. With the objective pair already
`-0.233/-0.623/+0.041`, this closes the exact temperature-0.03 route-E32 recipe as a reproducible
negative for a material or routing-mediated tail-safe benefit. Held-out fields are null, W&B sync
finishes, and no failure is excluded.

Container2874/GPU0 is immediately refilled with the highest-ranked remaining smoother row,
canonical-E64 temperature-0.1 tail-safe. Exact shard 6/8 exposes one unique pending cell; the GPU
has no compute PID, duplicate/result/prune/active guards pass, and the documented code-equivalent
execution base is `4893c964`. Controller66578 launches worker66583/W&B `7fnmaxhj`; startup reaches
8,855 MiB, loads 9,854 OOD-validation examples, explicitly leaves test untouched, and has no fatal
match. The only narrow repair writes the missing active marker once to the verified live worker PID
without restart. All ten allocated H100s again carry distinct workers. Smoother coverage becomes
7 active/1 queued of 8, extreme coverage becomes 4 completed/3 active/1 pruned of 8, ready falls
from 14 to 13, and backlog remains 27. Publication is not attempted because the already verified
private-repository storage quota blocker remains in force.

## 2026-08-02 23:54 EDT — Canonical E64 closes tail-adverse; final smoother row refills

The exact canonical-pressure E64 temperature-0.03 epoch-60 objective pair is strict-valid.
Tail-safe minus zero auxiliary is `+0.132/-0.150/-0.852` OOD-validation/ID/worst-experiment
points. The mean change is negligible, the worst-experiment effect is adverse, and randomized-
route reliance is `-0.00010` for tail-safe versus `0.00325` for zero auxiliary. This exact recipe
is terminal-negative for a material, tail-safe, or routing-mediated benefit; no epoch-90 or fresh-
seed continuation is licensed.

Both final JSONs match the declared canonical E64 seed-0 configuration, active-compute fairness
class, 96,865,524 total parameters, 1,206,145 active FFN-plus-router parameters, and epoch-60
checkpoint identity. Metrics are finite, all four environments and 9,854 OOD-validation examples
are present, controller exits are rc0, fatal scans are clear, W&B runs `cwzluit9` and `hkh3m1yq`
are synchronized, selection is `ood_val`, and every held-out field remains null. The result is
exploratory and multiplicity-exposed; generic regularization or trajectory noise is more plausible
than useful conditional routing.

Container2859/GPU0 is immediately refilled with the final temperature-0.1 registry row,
canonical-E64 zero auxiliary. Exact shard 7/8 exposes one unique pending cell; physical-GPU,
duplicate/result/prune/active, persistent-destination, tracking, checkpoint, and sealed-test guards
pass. Controller55069 launches worker55074/W&B `3g646u7z`; startup reaches 8,842 MiB, loads 9,854
OOD-validation samples with test untouched, and is fatal-clear. A missing active marker is repaired
once to verified worker55074 without restart.

Extreme temperature-0.03 coverage is now 5 completed, 2 active, and 1 pruned of 8. All 8 smoother
temperature-0.1 rows are active. Ten distinct workers occupy all ten running H100s; ready is 12 and
backlog 27. tester6/2899 remains scheduler Pending because 11 nodes fail affinity, two are
unschedulable, and three lack GPU capacity; preemption is unhelpful with no victim. HF publication
is withheld under the verified private-storage quota blocker. OOD test remains sealed.

## 2026-08-03 00:36 EDT — Extreme route E64 closes negative; E4 seed-1 refills submitted

The exact route-pressure E64 temperature-0.03 epoch-60 objective pair is strict-valid. Tail-safe
minus zero auxiliary is `-0.041/-0.468/-0.081` OOD-validation/ID/worst-experiment points. Route
reliance is `-0.00162` for tail-safe and `0.00020` for zero auxiliary, so neither result supports a
routing-mediated effect. Both controllers exited rc0, fatal scans are clear, the epoch-60
checkpoints and 10/30/60 milestones are present, all four environments and 9,854 OOD-validation
examples are covered, and every held-out field remains null. This exact seed-0 recipe is terminal-
negative for a material or tail-safe benefit and licenses neither epoch 90 nor fresh seeds.

The temperature-0.03 extreme family therefore closes with seven strict epoch-60 finals and one
preserved epoch-30 prune, with no active rows. Generic regularization or trajectory noise is more
plausible than useful conditional routing. The sharp falsifier is the smoother temperature-0.1
route-E32 final: it must retain aligned mean and tail movement and preferably exceed `0.01` route
reliance. The result remains exploratory and multiplicity-exposed after broad architecture,
pressure, bank-size, temperature, and objective searches.

The released container2887/GPU0 and container2859/GPU1 devices were strictly verified free before
submission. Locked canonical-E4 seed-1 tail-safe and zero-auxiliary confirmation commands were
then submitted with idempotency guards, using the exact direct code-equivalent configurations after
private-GitHub transport failed before mutation. Both direct configurations had already passed
one-cell dry runs and match data order, seed, schedule, active-compute fairness, parameter counts,
checkpoint policy, W&B group, HF destination, and sealed-test policy. Post-submission PID, GPU,
startup, and W&B verification is not claimed because SciServer terminal control degraded into a
bell-output flood. The last strict pool observation is eight active smoother workers and two free
devices before submission; current occupancy is unknown. Ready is conservatively ten, backlog is
27, and tester6/2899's last verified state remains Pending for the recorded scheduler constraints.

## 2026-08-03 00:49 EDT — SciServer table reconciliation; launch state remains unclaimed

The authoritative Compute table listed five running containers (`2887`, `2875`, `2874`, `2862`,
`2859`) and `tester6` container `2893` as stopped. It did not list the previously recorded
replacement `2899`. This corrects the operational registry; it does not change any scientific
result or authorize a duplicate container.

Exactly one start request was attempted for the stopped `2893`, as required. The request redirected
to a signed-out page and its outcome could not be verified. Chrome itself is running and its Codex
extension/native host pass local diagnostics, but the browser-control connection is unavailable;
the in-app session is unauthenticated. No claim is made that `2893` started or that either H100 is
available.

The two guarded E4 seed-1 submissions remain unverified. Pool accounting therefore stays at the
last strict pre-submission observation: eight smoother workers active and `2887/GPU0` plus
`2859/GPU1` free before submission. Current occupancy, launch PIDs, W&B state, milestones, and
fatal scans are unknown rather than inferred. Ready remains 10 and backlog 27.

There is no new scientific evidence. The extreme temperature-0.03 family remains closed at seven
strict finals and one valid prune; its route-E64 terminal contrast remains
`-0.041/-0.468/-0.081` OOD/ID/worst points with non-reliant routers. Generic regularization or
trajectory noise remains more plausible than reusable conditional routing; the smoother
route-E32 epoch-60 pair remains the sharp falsifier. Multiplicity is explicit and OOD test stays
sealed.

Next automatic action: restore an authenticated remote-control path, re-read the Compute table,
verify `2893` once, then inspect every running container separately. On `2887/GPU0` and
`2859/GPU1`, validate the submitted E4 cells or, only if absent and physically free, relaunch those
same guarded cells. Do not start or create another tester6 copy.

## 2026-08-03 01:06 EDT — tester6 is absent; replacement form prepared but not submitted

Authenticated SciServer control briefly recovered. The authoritative Compute table listed only
the five known 2H100 containers (`2887`, `2875`, `2874`, `2862`, `2859`), all with container-level
status `running`. Neither historical tester6 `2893` nor attempted replacement `2899` was present.
This supersedes the prior stopped-row observation for current container existence, but it does not
establish any per-GPU occupancy or worker health.

Under the pre-authorized create-once-if-absent rule, the new-container form was opened and verified
as `tester6`, `Interactive 2H100 GPU`, automatically selected `GPU-Essesials`, with persistent,
scratch, and data-vol mounted. Repeated Chrome-control disconnects occurred before the final Create
action, and the local computer-control fallback also failed to attach. Therefore zero create
submissions were made, no replacement ID exists, and no duplicate can have been created by this
invocation. The next invocation must first re-read the authoritative table; only if tester6 is still
absent may it submit exactly one replacement creation.

The per-container terminal audit could not proceed after the connection dropped. The E4 seed-1
submissions on `2887/GPU0` and `2859/GPU1` remain unverified. Pool accounting stays explicitly at
the last strict pre-submission observation: eight smoother workers and two then-free devices;
current active and idle counts are unknown. Ready remains 10 and backlog 27. No new result,
milestone, exclusion, promotion, publication, or scientific interpretation is recorded. The
extreme temperature-0.03 family remains terminal-negative for its exact recipes; generic
regularization or trajectory noise remains the leading explanation, smoother route-E32 epoch 60
remains the sharp falsifier, multiplicity remains explicit, and OOD test remains sealed.

## 2026-08-03 01:24 EDT — tester6 replacement reconciled and one start leaves it Pending

A fresh authenticated Compute-table read resolves the prior inconsistent existence observations:
`tester6` replacement container `2899` exists, was created at `2026-08-03 02:13:13.0`, and was
listed stopped before mutation. Its external container reference is
`e18249da-8ee0-11f1-a24e-0a580a8201b9`; historical container `2893` is not the current tester6.
The table also lists `2887`, `2875`, `2874`, `2862`, and `2859` running at container level.

Exactly one authorized start request was issued for `2899`; no creation request was issued. The
authoritative detail page then reported Kubernetes state `Pending`, `Running=false`, and exit code
zero. Its exact scheduling constraint is: 11 of 16 nodes fail affinity/selector, two nodes are
unschedulable, and three have insufficient `nvidia.com/gpu`; preemption is not helpful on 13 nodes
and finds no victim on three. Both tester6 H100s therefore remain unavailable, and no duplicate
start or container was created.

The five running containers could not be audited to completion: an attempted `2887` audit file was
not created, and repeated authenticated console-control timeouts prevented trustworthy PID, GPU,
epoch, log, checkpoint, fatal-scan, W&B, or HF refreshes. The E4 seed-1 submissions on
`2887/GPU0` and `2859/GPU1` therefore remain unverified. Current active-worker and idle-H100 counts
are unknown; the last strict pre-submission map remains eight workers and two then-free devices,
and is not relabeled as current occupancy. Ready remains 10 and backlog 27.

No scientific evidence moved. The extreme temperature-0.03 family remains terminal-negative for
its exact recipes; generic regularization or trajectory noise remains more plausible than useful
conditional routing. Smoother route-E32 epoch 60 remains the sharpest falsifier, the evidence is
still exploratory and multiplicity-exposed, and OOD test remains sealed. Next automatic action is
to re-read `2899` once, audit all five running containers separately, and validate the two E4
seed-1 cells or relaunch only the same guarded cells if absent and physically free. Start `2899`
again only if a later authoritative table shows it stopped; do not create another tester6.

## 2026-08-03 02:02 EDT — smoother route-E64 closes negative; confirmation test gate repaired

The smoother temperature-0.1 route-E64 tail-safe and zero-auxiliary epoch-60 finals now form a
strict seed-0 active-compute pair. Tail-safe minus zero auxiliary is `-0.041/+0.059/-0.325`
OOD-validation/ID/worst-experiment points. Both runs cover four environments and 9,854 validation
examples, preserve nonempty epoch-10/30/60 checkpoints, select on `ood_val`, leave test unevaluated
with held-out fields null, and use identical 96,865,524 total and 1,206,145 active FFN-plus-router
parameters. Route reliance is `-0.00020` versus `0.00325`; this is terminal-negative for a
material, tail-safe, or routing-mediated benefit in the exact recipe. No epoch 90 or fresh seed is
licensed. Generic regularization or trajectory noise is more plausible, subject to the explicit
multiplicity caveat after broad architecture, pressure, bank-size, temperature, and objective
searches. The strict record is
`analysis/smoother_route_E64_epoch60_paired_validation.json`.

The prior E4 confirmation submissions are absent on the devices that could be checked. A clean
SciServer clone of local commit `867788d` exposed one test-only parser error: tags containing
`tail_safe` or `no_aux` were split into too many fields. The narrow `split("_", 2)` repair is
committed and pushed as `80671979e2bfc88f3e8b6da4aa02bb67ae02be3e`; its complete bundle is in a
new clean SciServer checkout. Focused and full test execution was submitted there, but completion
could not be read after the authenticated browser session was interrupted. No E4 confirmation run
is claimed or launched under this unverified test gate.

The strict partial GPU map is: `2887/GPU0` and `2887/GPU1` idle at 0 MiB; `2862/GPU0` occupied by
PID 66583 at 8,867 MiB and `2862/GPU1` idle at 0 MiB; `2859/GPU0` idle at 0 MiB and `2859/GPU1`
occupied by PID 121339 at 8,407 MiB. Containers `2874` and `2875` remain table-level running but
were not audited per GPU, so pool-wide active and idle counts are unknown; four H100s are strictly
verified idle, not a saturation claim. Tester6/2899 remains Pending after the single start request
with 11 affinity mismatches, two unschedulable nodes, three insufficient-GPU nodes, and no useful
preemption. Ready/backlog remain conservatively 10/27. The next action is to read the repaired test
report; only if focused and full suites pass, run exact dry-runs and launch the locked E4 seed-1
tail-safe/zero-auxiliary pair on verified-free devices with duplicate guards, then fill the next
verified releases with the predeclared seed-2 pair. OOD test remains sealed.

## 2026-08-03 02:21 EDT — repaired confirmation source passes focused and full local suites

An isolated macOS/arm64 Python 3.9 environment independently reran the repaired source at commit
`80671979e2bfc88f3e8b6da4aa02bb67ae02be3e`. The three focused registry/refill files pass 18/18,
and the full suite passes 153/153 in 90.35 seconds. The ten warnings are Pillow `mode` deprecations
only. This closes the source-level test gate that previously failed solely because the test parser
split `tail_safe` and `no_aux` labels at every underscore. The validation record is
`analysis/expert_count_aux_confirm60_local_test_validation.json`.

This is correctness evidence, not scientific evidence or a launch. The previous SciServer
pre-repair suite already passed 152 tests with only that deterministic parser failure; the submitted
post-repair remote report remains unread while the user reauthenticates SciServer. The locked E4
seed-1 and seed-2 auxiliary pairs are now source-test licensed, but each launch still requires a
fresh authoritative container/GPU audit, duplicate/result/marker checks, the exact one-cell remote
dry-run, persistent destinations, and sealed-test verification. No current occupancy count is
claimed from the earlier partial snapshot. Registered ready/backlog remain 12/27. Browser control
is intentionally paused for user reauthentication; OOD test remains sealed.

## 2026-08-03 03:12 EDT — canonical-E64 smoother pair closes negative; ten H100s refill cleanly

The canonical-pressure E64 temperature-0.1 epoch-60 pair is now strict-valid. Tail-safe minus exact
zero auxiliary is `-0.477/+0.192/+0.203` OOD-validation/ID/worst-experiment points. Route reliance
is `-0.000812` versus `0.000609`, so the small tail movement is not evidence of conditional
routing. Both finals are finite, cover four environments and 9,854 validation examples, have
identical 96,865,524 total and 1,206,145 active FFN-plus-router parameters, preserve epoch-10/30/60
checkpoints, synchronize W&B, select only on `ood_val`, leave test unevaluated, and keep held-out
fields null. This is a terminal seed-0 active-compute negative for a material or routing-mediated
benefit in the exact recipe; neither epoch 90 nor fresh seeds is licensed. The record is
`analysis/extreme_temperature_aux60_canonical_E64_temp01_epoch60_paired_validation.json`.

The repaired remote gate is also complete: commit `80671979e2bfc88f3e8b6da4aa02bb67ae02be3e`
passes 18 focused and 153 full SciServer tests. Exact one-cell dry-runs for all twelve then-ready
cells report one planned and one pending cell. The five running worker containers were separately
audited before and after launch. All ten available H100s now run distinct owned jobs with fresh
logs, exact active PIDs, W&B IDs/groups, 7,999–8,875 MiB allocations, and the explicit 9,854-sample
OOD-validation/test-untouched startup line. Container/GPU allocation is: `2887` E4 seed-1
tail-safe/zero-aux; `2875` E4 seed-2 tail-safe/zero-aux; `2874` route-E32 temperature-0.3
tail-safe/zero-aux; `2862` route-E64 temperature-0.3 tail-safe/zero-aux; and `2859` canonical-E32
temperature-0.3 tail-safe/zero-aux. There are ten active workers and zero idle available H100s.

Tester6 replacement `2899` was re-read after the launches and remains `Pending`, `Running=false`,
exit zero with the unchanged scheduler reason: 11 nodes fail affinity/selector, two are
unschedulable, three have insufficient GPU, and preemption is unhelpful with no victims. This
invocation issued exactly one start and zero creates; no retry or duplicate is licensed while it is
Pending.

Queue depth is restored by predeclaring the twelve-cell E4/E8/E16 route/canonical x
tail-safe/zero-auxiliary temperature-0.3 family. Nine focused and 156 full local tests pass, and
boundary shards 0 and 11 each dry-run as one unique pending cell. Together with the two still
queued canonical-E64 temperature-0.3 controls, ready is 14 and backlog remains 27. The new family
still requires transfer, remote focused/full validation, and exact remote per-shard dry-runs before
launch. The first physical release goes to canonical-E64 temperature-0.3 tail-safe, then its exact
zero-auxiliary comparator. HF publication remains deferred under the known private-storage quota
blocker; OOD test remains sealed.

## 2026-08-03 04:42 EDT — twenty locked milestones validate; route-E32 temperature-0.3 leads provisionally

Twenty epoch-10/30 rows from the ten active workers are strict-valid. Every row is finite, exact
for run/seed/epoch/config, covers environments 7/27/42/49 with 9,854 OOD-validation examples,
records `selection_split=ood_val` and `test_evaluated=false`, and names a nonempty milestone
checkpoint. All twenty checkpoints are present at their pair-specific expected sizes; the current
campaign fatal scan is zero bytes. The clean execution checkout is commit
`80671979e2bfc88f3e8b6da4aa02bb67ae02be3e`. The validation record is
`analysis/temperature03_and_E4_confirmation_epoch10_30_validation.json`.

The strongest interim pair is route-pressure E32 at temperature 0.3: tail-safe minus exact
zero-auxiliary changes from `-0.284/-0.490/-0.122` OOD/ID/worst points at epoch 10 to
`+1.644/+1.573/+0.041` at epoch 30. This is an exploratory trajectory below the +5 material gate,
not a promotion. Route-E64 at epoch 30 is `-0.690/+1.115/+0.081`; canonical-E32 is
`+0.497/-0.790/-0.365`. Both remain Pareto tradeoffs.

The locked E4 fresh seeds disagree at epoch 30. Seed 1 tail-safe minus zero auxiliary is
`+0.233/-0.549/-0.122` OOD/ID/worst points, while seed 2 is
`-0.771/-0.726/+0.000`. The seed-0 E4 auxiliary signal is therefore not reproduced at this
interim milestone. All four locked rows continue unchanged to epoch 60 because the final mechanism
signature and final confirmation milestone were predeclared; no epoch-90 branch is licensed.

All five worker containers were audited independently. Containers 2887, 2875, 2874, 2862, and
2859 each retain two distinct owned workers with 7,999--8,903 MiB allocations; current logs range
through epoch 30--43. Ten workers occupy the ten available H100s and zero are idle. Instantaneous
zero-utilization samples occurred between kernels/evaluations but each corresponding worker and
GPU allocation remained present. Tester6/2899 remains `Pending`, `Running=false`, exit zero for
the unchanged 11 affinity mismatches, two unschedulable nodes, three insufficient-GPU nodes, and
unhelpful preemption/no victims. No start or create was issued this invocation.

Ready/backlog stay 14/27. The first physical release still launches canonical-E64 temperature-0.3
tail-safe, followed by its exact zero-auxiliary comparator. The moderate-bank family remains
release-ineligible until clean remote transfer, focused/full tests, and exact one-cell dry-runs
pass. Generic optimization or regularization trajectory remains more plausible than conditional
routing; route-E32 must retain aligned mean/tail movement and show non-negligible final route
reliance to falsify that explanation. Multiplicity is explicit, HF remains quota-blocked, and OOD
test is sealed.

## 2026-08-03 06:19 EDT — eight terminal pairs/confirmations close negative; ten GPUs refill and queue returns to 13

Eight paired epoch-60 finals and one additional unpaired canonical-E32 level pass the strict
validator. All required metrics are finite, the eight paired rows have identical seed/config/data
and parameter identity within pair, all nine cover environments 7/27/42/49 and 9,854 OOD-validation
samples, all 24 required pair checkpoints exist, fatal scans are empty, selection is `ood_val`,
`test_evaluated=false`, and held-out fields remain null. The record is
`analysis/temperature03_epoch60_and_refill_validation.json`.

The locked E4 fresh-seed confirmation is terminal-negative for a useful auxiliary effect. Tail-safe
minus exact zero auxiliary is `+0.020/-0.108/-0.122` OOD/ID/worst points for seed 1 and
`+0.213/+0.175/-0.487` for seed 2. Both tiny mean signs lose worst-environment accuracy, and route
reliance stays between roughly `-0.002` and `0.004`. This rules out the seed-0 tail-safe signal as
a reproducible sparse-specialization or tail-safe effect for the exact recipe; no epoch 90 or more
seeds are licensed.

The temperature-0.3 route-pressure pairs also close negative. E32 tail-safe minus no auxiliary is
`+0.304/-0.187/-0.446`, and E64 is `+0.102/+0.468/-0.244` OOD/ID/worst points. Reliance remains
near zero. The promising E32 epoch-30 movement therefore did not persist to epoch 60. These exact
E32/E64 recipes license neither epoch 90 nor fresh seeds. A generic regularization or trajectory
explanation is now more plausible than useful conditional routing; the sharp falsifier remains an
aligned mean-and-tail epoch-60 pair with route reliance above `0.01`, followed by locked seeds.

Every completion handed off immediately. Container `2887` now runs the canonical-E64
temperature-0.3 tail/no-aux pair; `2875` route-E4 tail/no-aux; `2874` route-E8 tail/no-aux; `2862`
route-E16 tail/no-aux; and `2859` canonical-E32 tail plus canonical-E4 tail. Ten distinct workers
occupy all ten H100s in the five running containers, leaving zero idle available devices.
Tester6/2899 remains `Pending`, `Running=false`, exit zero for the unchanged scheduler reason; no
start or create was issued. The canonical-E32 no-aux final is strict-valid but remains unpaired
until its active tail-safe counterpart finishes.

The moderate bank passes its isolated remote gate and has seven active plus five queued arms.
Queue depth is replenished by commit `3eca8fc27e16a2f7e4126c22bfa491b5231b9227`, which predeclares
eight temperature-0.3 noise-upcycling cells against already-running zero-noise anchors. The exact
remote files pass three focused and 130 full tests, and all eight shards dry-run as one unique
pending cell. Ready/backlog are 13/26. On the first release, launch canonical-E4 no-aux, then the
canonical E8/E16 pairs; after those, consume the noise queue in registry rank order under fresh
GPU/duplicate/result/marker guards. W&B is live, HF remains quota-blocked, multiplicity is explicit,
and OOD test remains sealed.

## 2026-08-03 09:21 EDT — four temperature-0.3 pairs close negative; nine releases refill immediately

Four completed epoch-60 pairs and one unpaired canonical-E4 tail-safe final pass the strict
validator. The paired tail-safe-minus-no-auxiliary OOD/ID/worst changes are
`+0.375/+0.347/+0.122` for canonical E64, `+0.660/+0.409/-0.122` for route E4,
`+0.700/+0.379/-0.487` for route E8, and `+0.396/+0.534/-0.162` for route E16. All
route-reliance magnitudes remain below `0.005`. These exact seed-0 recipes are terminal-negative
for a material or routing-mediated tail-safe benefit and license neither epoch 90 nor fresh seeds.
Canonical E4 tail-safe is strict-valid at `0.20692/0.52411/0.01705` OOD/ID/worst, but its
pairwise effect remains withheld until the active no-auxiliary row finishes. The complete record is
`analysis/temperature03_moderate_epoch60_and_upcycling_refill_validation.json`.

Every released device was refilled after an exact one-cell dry-run. Container 2887 runs the
canonical-E8 tail/no-aux pair, 2875 the canonical-E16 pair, 2874 the route/canonical E4
noise-0.01 pair, 2862 the route/canonical E8 noise-0.001 pair, and 2859 canonical-E4 no-aux plus
route-E8 noise-0.01. Ten distinct workers occupy the ten available H100s and zero are idle. A
missing active marker was restored once for each verified live worker without a restart. New logs
are fatal-clear, W&B is initialized, and startup remains OOD-validation-only with test untouched.

Five moderate-bank rows remain active and seven are final; five upcycling rows are active and
three remain ready. A 14-arm learned-router/frozen-router/dense-noise mechanism-control family is
predeclared at commit `04ad526`, with local syntax, JSON, and diff checks passing. Its remote
transfer, focused/full tests, and 14 exact dry-runs remain a launch gate because Chrome control
dropped after the transfer attempt; those 14 are not counted runnable. The pool is nevertheless
fully assigned. Tester6/2899 remains Pending for the unchanged scheduler reason, with no start or
create issued. Generic regularization or trajectory noise is the leading explanation; the sharp
falsifier remains aligned mean-and-tail improvement, route reliance above `0.01`, and locked-seed
survival. HF is quota-blocked, multiplicity is explicit, and OOD test remains sealed.
## 2026-08-03 14:02 EDT — nine workers remain active; dense mechanism control receives a narrow Stage-0 repair

Authenticated SciServer access was working and supported a complete fresh audit before the browser-control
transport disconnected. Five upcycling jobs had completed rather than disappeared during container restarts;
their persistent results are present but remain unvalidated and support no scientific claim yet. Immediate
handoffs launched upcycling ranks 6--8 on `2875/gpu1` and both GPUs of `2874`, resumed the guarded canonical-E4
no-auxiliary comparator on `2862/gpu0`, and launched frozen-router controls on `2862/gpu1` and `2859/gpu1`.
At the last strict audit, nine of ten H100s in the five allocated worker containers had distinct active workers.

The first dense noise control on `2859/gpu0` failed before training during Stage-0 model construction because
the predeclared script used the obsolete token `wide`; the accepted exact variant is `dense_wide`. No result was
created, so this is an implementation failure rather than scientific evidence. The script and its registry test
were repaired narrowly, transferred with exact hashes, and pass local syntax and diff checks. Before the failure,
the remote gate had passed three focused tests, 133 full tests, and all fourteen exact one-cell planning dry-runs.
The post-repair remote test and shard-4 readback was not observed because the Chrome control transport reset and
then disconnected. The repair record is `analysis/upcycling_noise_controls60_dense_variant_repair.json`.

Container `2859/gpu0` is the one known idle allocated GPU. Its exact retry is licensed only after authenticated
browser control returns, the repaired focused/full test output and shard-4 one-pending dry-run are observed, and
fresh free-GPU/result/marker/duplicate guards pass. Tester6/2899 appeared stopped with a Start Container link on
the latest Compute table and contributed no GPUs; it must be rechecked and started exactly once if still stopped,
never recreated. Six unaffected frozen/learned controls remain runnable; the six dense controls are test-gated.
OOD test remains sealed, HF publication remains quota-blocked, and the paper is unchanged.

## 2026-08-03 14:39 EDT — in-app SciServer transport works; one-time in-app login is required

The in-app browser reached the BNL SciServer login portal without a transport timeout. Its browser
profile is separate from Chrome and is not authenticated, so the Compute and Jupyter audit cannot
continue until the user completes the federated login once in the in-app tab now left open. No
container, job, result, publication, or scientific state changed. The last strict pool snapshot
therefore remains nine active workers and one known idle allocated GPU on `2859/gpu0`; tester6/2899
last appeared stopped and contributes zero. After login, the immediate licensed sequence is to
read the repaired remote test/dry-run gate, refill `2859/gpu0`, validate five completed upcycling
results, and inspect/start tester6 exactly once only if still stopped. OOD test remains sealed.

## 2026-08-03 16:01 EDT — authentication works; five upcycling finals validate and the repaired control fills the idle H100

The in-app SciServer session is authenticated and can reach the Compute dashboard and all five worker
containers. A fresh per-GPU audit found ten distinct live workers across `2887`, `2875`, `2874`, `2862`, and
`2859`; zero of the ten currently available H100s are idle. Tester6/2899 received exactly one Start request and
remains scheduler-Pending because no compatible capacity is available: 11 nodes fail its affinity/selector, two
are unschedulable, and three lack sufficient GPUs. It contributes zero GPUs and was not restarted or recreated.

The repaired dense-control implementation passes three focused tests, all 133 repository tests, and the exact
shard-4 dry-run with one planned/one pending cell. Fresh checks showed `2859/gpu0` owned and empty with no
duplicate process or result. The failed cell was retried exactly once as controller 2458/worker 2478 and verified
live at 8,831 MiB and 99% utilization under the predeclared controls campaign.

Five seed-0 upcycling epoch-60 results pass strict result, environment, heldout-field, log, provenance, W&B-ID,
and epoch-10/30/60 checkpoint checks. All have absolute route reliance below `0.0017`, far below the predeclared
`0.01` specialization signal. Against exact zero-noise comparators, canonical E4 noise 0.01 moves OOD/ID/worst
by `+0.132/+0.561/+0.162` points; route E4 noise 0.01 moves `-0.558/+0.195/-0.528`; route E8 noise 0.001 moves
`+0.721/-0.537/+0.203`; and route E8 noise 0.01 moves `+0.304/-0.389/+0.041`. The canonical E8 noise-0.001
result remains unpaired until its zero-noise comparator finishes. These small mixed exploratory effects do not
show useful conditional routing and do not justify epoch 90 or fresh seeds. Ordinary initialization/trajectory
regularization is the leading explanation. Twelve distinct cells remain ready for immediate refill. HF remains
quota-blocked, the paper is unchanged, multiplicity is explicit, and OOD test remains sealed.

## 2026-08-03 18:04 EDT — dense E4 control validates at epoch 30; all ten workers continue

The repaired exact-total-parameter-matched dense E4 noise-0.01 control now has a strict-valid
epoch-30 milestone and a 297.3 MB checkpoint. Metrics are 17.719% OOD validation, 46.016% ID, and
1.420% on the worst held-out experiment. The row is finite ERM, matches the exact run/seed/epoch,
covers environments 7/27/42/49 with 9,854 samples, uses `selection_split=ood_val`, records
`test_evaluated=false`, and has no fatal-log signature. Its controller has not exited and its fresh
run log has advanced through epoch 38. The validation record is
`analysis/upcycling_noise_controls60_dense_E4_epoch30_validation.json`.

At the same milestone, the learned E4 noisy router exceeds dense-wide by
`+1.806/+0.086/+0.649` OOD/ID/worst-experiment points and exceeds the frozen-router control by
`+0.538/-0.544/+0.365`. This is an encouraging aligned interim difference against dense, but it is
well below five points, the learned arm's final route reliance is only `0.00142`, and the dense
comparator is unfinished. It therefore does not establish useful conditional routing and licenses
no epoch-90 or fresh-seed run. Ordinary initialization or trajectory effects remain the leading
explanation until the exact epoch-60 comparison closes.

All ten available workers remain assigned with fresh persistent logs and no controller exits:
the E8/E16 zero-noise comparators are at epoch 25, six newly launched E8/E16 controls are at epoch
28, dense E4 is at epoch 38, and dense E16 is at epoch 28. This maps to two workers each on
2887/2875/2874/2862/2859 and zero idle available H100s. The browser terminal keyboard bridge did
not execute a fresh `nvidia-smi`, so current liveness is based on per-container controller identity
plus minute-fresh trainlogs; prior exact memory figures are not relabelled as current. Tester6/2899
remains Pending for the unchanged scheduler reason, with no start or create issued. Ready/backlog
remain 12/26, W&B streams are live, HF remains quota-blocked, the paper is unchanged, and the OOD
test remains sealed.

## 2026-08-03 21:10 EDT — gradient localization validated; ten two-epoch paired screens launched

The training-only all-layer profiler completed all 10 declared profiles: pretraining plus epochs
10/30/60/90, each with two independent gradient draws, 12 FFNs, 33 training experiments,
8 samples per experiment, 3 rounds, and a 4,096-dimensional deterministic sketch. Every JSON is
parseable and finite, the fatal scan is clear, and the profiler records no selection split,
`test_evaluated=false`, and no held-out accuracy fields. Block 11 is the reproducible conflict peak
at epochs 10 and 30 and remains first or second at epoch 60; blocks 10+11 form the high-conflict
two-FFN intervention; block 1 is the low-conflict placebo at epochs 30 and 60. Epoch-90 conflict is
nearly saturated across all layers and is not a useful placement discriminator. This is diagnostic
mechanism evidence, not an accuracy improvement. Trace:
`analysis/gradient_conflict_profile_validation.json`.

Ten disjoint epoch-2 arms now occupy all 10 schedulable H100s across containers
2887/2875/2874/2862/2859: learned, frozen-router, and exact-total-parameter-matched dense controls
at block 11; the same triplet at placebo block 1; learned/frozen/dense controls at blocks 10+11;
and an original-width Cell-DINO anchor. The single-block sparse/dense totals are
30,676,212/30,675,834 (378 parameters apart); the expected two-block totals are
38,950,261/38,949,505 (756 apart). All use seed 0, the same data order and optimizer, save an
epoch-2 checkpoint, select only on OOD validation, and keep OOD test sealed. Initial processes are
live with GPU memory allocated and no fatal signatures; no accuracy result is yet available.

Tester6 replacement 2899 received exactly one start request and remains scheduler-Pending with
zero available GPUs: 11 nodes fail affinity/selector, 2 are unschedulable, and 3 lack GPUs;
preemption is unhelpful or has no victims. Thus the usable pool is 10/10 occupied, with zero idle
usable GPUs and two authorized GPUs currently unavailable. Twelve fully specified 5/10-epoch
follow-ups and 24 mechanism hypotheses are ready in
`analysis/fast_conflict_screen_registry.json`. The next completed epoch-2 triplet is validated as a
pair before its leading family receives epoch 5; no broad 60- or 90-epoch search will be restarted.

## 2026-08-03 23:18 EDT — layer-dispatch failure preserved, repaired, and ten corrected epoch-2 searches launched

The first epoch-2 screen completed, but strict result inspection found that all nine non-original
rows recorded `block_indices=[6]` and `n_blocks_converted=1`. The launch registry requested block
11, block 1, or blocks 10+11 through `model.ffn_block_indices`, while end-to-end construction still
read the older `model.block_indices` key and silently used the legacy middle placement. Those nine
rows are preserved as actual block-6 observations but excluded from every placement, placebo, and
one-versus-two-FFN comparison. Their 0.812--1.076% OOD-validation levels and near-zero randomized-
route differences are too early and cannot answer the intended question. The original-width epoch-2
anchor is unaffected. Trace: `analysis/fast_conflict_screen_epoch2_dispatch_failure.json`.

GitHub commit `813015d2309cec6a49bbc886c18230e5b26b502c` makes naming and model construction share one
fail-closed block-index resolver. The clean code-equivalent SciServer commit
`3b898c2fe0c3445ca65826a2b5a692aae475e143` passes 37 focused and 109 full tests. Twelve exact
preflight builds confirmed blocks 11, 1, 10+11, and 10, unique run IDs, and the expected one-/two-
FFN parameter totals. Three legacy epoch-60 holders with no log advance for about 12 hours were
recorded, stopped, and not restarted.

Ten corrected epoch-2 processes now occupy all ten available H100s: the learned/frozen/dense block-11
triplet; learned/frozen/dense blocks-10+11 triplet; learned/frozen/dense block-1 placebo triplet; and
a learned block-10 localization follow-up. Block-10 frozen and dense controls are the first two
refills. All use seed 0, matched data order, a fresh W&B group, persistent results/checkpoints,
OOD-validation selection, and sealed OOD test. No corrected accuracy result exists yet. Trace:
`analysis/fast_conflict_screen_dispatch_repair_validation.json`.

## 2026-08-04 00:18 EDT — ten corrected epoch-2 results validate and the epoch-5 screen fills the pool

All ten corrected results pass strict JSON, finite metric, exact block/variant, parameter-count,
four-environment/9,854-sample, checkpoint, milestone, log, manifest, provenance, and fatal-scan
checks. Every result uses `selection_split=ood_val`, records `test_evaluated=false`, and keeps all
five OOD-test fields null after a fail-closed metadata normalization. Ten raw pre-normalization
copies are preserved, ten manifests were regenerated and hash-verified, and no scientific value
changed. Seven sparse/frozen rows use all eight experts with routing entropy 0.9598--0.9968.
Trace: `analysis/fast_conflict_dispatchfix_epoch2_validation.json`.

At this very early point, block-11 learned routing is `1.16%` OOD validation versus `1.07%` dense
and `0.87%` frozen; blocks 10+11 learned is `1.09%` versus `1.05%` dense and `0.99%` frozen. The
block-1 placebo learned row is `0.91%`, below its `1.02%` dense control but above `0.79%` frozen.
This is the predicted localization ordering, but the accuracies are near chance and the differences
are only diagnostic. The sharp falsifier is failure to preserve or strengthen learned-over-control
ordering at epoch 5, or seeing an equal effect at the placebo.

Eight epoch-5 workers and the two missing block-10 epoch-2 controls now occupy all ten schedulable
H100s. Exact assignments are 2887: high11 learned/frozen epoch 5; 2875: high10 frozen epoch 2 and
high11 dense epoch 5; 2874: high10 dense epoch 2 and high10+11 learned epoch 5; 2862: high10+11
frozen/dense epoch 5; 2859: low1 learned/dense epoch 5. `low1_frozen_ep5` is the immediate refill.
The ready queue remains at least 12 with 24 hypotheses in backlog. tester6/2899 received one start
and remains Pending because available nodes do not satisfy affinity/capacity; no duplicate was
created or started.

One bounded HF publication attempt stopped immediately at the private-repository storage limit;
zero normalized runs uploaded and no retry will occur until quota is available. The source writer
now normalizes unevaluated OOD-test fields before persistence. Local syntax and extracted-helper
tests pass; a new full remote suite was blocked by the SciServer checkout's GitHub reauthentication
prompt, so no credential was entered. Evidence, registries, source repair, and ledgers were pushed
in GitHub commit `5c305f8696e4a7a823d6a32ae07b0f6b66a47cb3`. OOD test remains sealed and the
manuscript is unchanged.

## 2026-08-04 01:17 EDT — two conflict-localized FFNs show the first promising short signal; ten follow-ups launch

Eight epoch-5 rows and the two block-10 epoch-2 controls are strict-valid. Every row parses, is
finite, uses the exact requested blocks and parameter class, covers environments 7/27/42/49 with
9,854 OOD-validation samples, has its required checkpoint/log/manifest companions, is fatal-clear,
records `selection_split=ood_val` and `test_evaluated=false`, and leaves OOD test sealed. The two
block-10 controls had written to the persistent default results directory because their launch
commands omitted `--results-dir`; their existing files were copied without overwrite into the
declared campaign root. No scientific value changed. Trace:
`analysis/fast_conflict_epoch5_validation.json`.

The strongest epoch-5 comparison is the two-FFN intervention at blocks 10+11. Learned sparse is
`2.994%` OOD validation versus `2.882%` frozen sparse and `1.796%` equal-total-parameter dense:
learned-minus-dense is `+1.197` points and learned-minus-frozen is only `+0.112` points. Its worst-
environment advantage over dense is `+0.446` points. In contrast, block-11 learned is `2.405%`
versus `2.639%` dense, and the block-1 placebo learned is `1.705%` versus `1.938%` dense. All sparse
rows use 8/8 experts with entropy at least `0.996`, but randomized-route reliance is only
`0.00030--0.00213`. The most plausible current explanation is therefore a useful two-layer expert
partition or structured regularization effect, with only weak evidence that the learned router is
doing more than a fixed partition. This is exploratory seed-0 evidence after multiple comparisons,
not a confirmed performance claim.

All ten schedulable H100s were free after the short batch and were immediately refilled. Exact
assignments are: 2887 GPU0/1 `low1_frozen_ep5` / `high10_learned_ep5`; 2875 GPU0/1
`high10_frozen_ep5` / `high10_dense_ep5`; 2874 GPU0/1 blocks10+11 learned/frozen epoch10; 2862
GPU0/1 blocks10+11 dense epoch10 / block11 learned epoch10; 2859 GPU0/1 block11 frozen/dense
epoch10. Each GPU has one owned Python compute process and roughly 7.9--11.8 GB allocated; startup
logs identify 33 training experiments, 9,854 OOD-validation images, a fresh W&B group, and explicit
test blindness. Empty PID markers caused by outer-shell expansion were repaired from the live GPU
process table. Trace: `analysis/fast_conflict_continuation_registry.json`.

The continuation passed ten unique config/run-ID checks, five representative exact model builds,
and a finite `2x1139` real-data forward dry run after one narrow batch-unpacking repair. A separate
12-arm ready queue of bounded load-balance, temperature, symmetry-breaking, top-2, and image-router
screens also passes unique-ID and representative build preflight. The next release takes the
highest-ranked nonduplicate arm from `analysis/fast_conflict_ready_queue_registry.json`. No HF
retry was made because the private storage quota is unchanged; the paper remains unchanged.
Validated evidence and launch registries were pushed in GitHub commit
`f4ff4b5ab3ef52cd3336801487175e0defdf763b`.

## 2026-08-04 02:37 EDT — epoch-10 milestone signal, writer repair, and fast refill

All ten continuation jobs reached their declared terminal milestone and wrote nonzero checkpoints,
but the old post-training writer then failed with a missing-helper `NameError`. Their full result
JSONs are excluded until recovered; training is not being repeated. Source commit `c68bc80` adds
the missing normalization helper and a fail-closed checkpoint finalizer. Its code-equivalent
SciServer commit `818fc8a` passes 35 focused and 114 full tests. Four key checkpoints are being
finalized now and six remain queued. Trace:
`analysis/post_training_finalizer_repair_validation.json`.

The strict milestone values change the mechanism interpretation. Blocks10+11 learned/frozen/dense
at epoch 10 score `7.083/7.104/5.845%` OOD validation. The sparse-over-dense contrast is therefore
`+1.238` points, but learned routing is `-0.020` points versus frozen routing. Block11 learned is
`-0.944` points versus dense, block10 learned at epoch 5 is `-0.649` points, and the low-conflict
frozen placebo is `+0.781` points versus dense. The two-FFN effect persists, but the evidence now
favors fixed partitions, conditional capacity, or optimization regularization rather than useful
adaptive routing or conflict-specific placement. Trace:
`analysis/fast_conflict_epoch10_milestone_validation.json`.

The ten freed H100s were first refilled with six new five-epoch performance searches and four
checkpoint recoveries. Exact performance assignments were 2887 GPU0/1 auxiliary weights 0 and 0.0001; 2875
GPU0/1 symmetry noise 0.0001 and 0.001; 2874 GPU0/1 router temperatures 0.03 and 0.15. Container
2862 recovered the two-FFN learned/frozen epoch-10 finals; 2859 recovered the matched dense final
and block-1 frozen placebo. All four recovered results are strict-valid, clean `818fc8a`, and were
not retrained. Four checksum manifests cover result/trainlog/milestones/checkpoint and reproduce
the recorded checkpoint hashes. Learned/frozen route reliance is only `0.00721/0.00680`, both below `0.01`, so the
full mechanism results reinforce the fixed-partition interpretation. Those four released GPUs were
immediately refilled with auxiliary-weight 0.05/0.1 and the learned/frozen top-2 pair. All ten GPUs
again run short performance searches. The ready queue remains at 12 after exact-building E4/E16
top-1 triplets and top-2 pairs; 24 backlog hypotheses remain. OOD test is sealed, tester6 remains
scheduler-Pending, HF waits for private quota, and the paper is unchanged. Trace:
`analysis/fast_conflict_ready_queue_extension_validation.json` and
`analysis/fast_conflict_recovered_finals_validation.json`.

## 2026-08-04 03:24 EDT — first router-dynamics wave validates and four releases refill

- **Where we are:** four of the first ten router-dynamics screens are strict-valid. Ten H100
  assignments remain occupied and zero usable H100s are idle: 2887 runs auxiliary weights 0 and
  0.0001; 2862 runs 0.05 and 0.1; 2859 runs learned/frozen top-2; 2874 was refilled with temperature
  0.30 and image routing; 2875 was refilled with the paired E4 learned/frozen models. Twelve exact-
  build-checked arms remain ready and 24 hypotheses remain in backlog. tester6/2899 is still
  Pending because 11 nodes fail affinity, two are unschedulable, and three lack GPU capacity.
- **What moved:** temperature 0.03/0.15 and symmetry-noise 0.0001/0.001 completed with two checkpoints
  each, valid epoch-2/5 streams, clean `818fc8a` provenance, four-environment coverage, null heldout
  fields, and fatal-clear logs. The four released GPUs were immediately refilled. Four linear-router
  learned/frozen top-1/top-2 arms were exact-built at the same blocks 10+11 and restored ready depth
  to 12.
- **What we learned:** temperature 0.03 and 0.15 differ negligibly from the prior two-FFN learned
  reference. Noise 0.001 is the short family leader at `3.258%` OOD validation, `4.415%` ID, and
  `0.974%` worst environment. Relative to the prior learned reference this is `+0.264/+0.443/-0.162`
  points. Its route reliance is only `0.00244`, so the result currently favors initialization or
  optimization regularization, not learned conditional routing. The sharp falsifier is a matched
  frozen result that matches learned, or continued reliance below `0.01`.
- **Correctness/trust:** all four results are exploratory seed 0 within a multiple-comparison search.
  Parameter accounting is `38,950,261` total and `2,369,282` active FFN parameters; all eight
  experts are used and entropy is at least `0.99917`. Selection is OOD validation only; OOD test is
  sealed. The largest threat is multiplicity plus the small absolute effect and lower tail metric.
- **Traceable artifacts:** `analysis/fast_conflict_router_dynamics_epoch5_wave1_validation.json`,
  `analysis/fast_conflict_ready_queue_linear_extension_validation.json`, and the updated ready
  registry and steward ledgers. HF remains quota-blocked; the manuscript is unchanged.
- **Next automatic action:** validate the pending auxiliary/top-2 releases, put the E4 dense control
  on the next free GPU, then test learned versus frozen within E4. Continue refilling from the
  twelve-arm queue; no new broad 60/90-epoch search is authorized.

## 2026-08-04 03:52 EDT — balance strength and E8 top-2 close; six GPUs refill with paired controls

- **Where we are:** six additional epoch-5 rows are strict-valid. All ten schedulable H100
  assignments were refilled: 2887 runs E4 dense and E16 learned; 2875 runs E4 learned/frozen;
  2874 runs temperature 0.30 and image routing; 2862 runs E16 frozen/dense; and 2859 runs E4
  top-2 learned/frozen. Zero usable H100 assignments are idle. The exact-build ready queue is 12
  and the hypothesis backlog is 24. tester6/2899 remains scheduler-Pending and contributes no GPU.
- **What moved:** auxiliary weights 0, 0.0001, 0.05, and 0.1, plus E8 top-2 learned/frozen,
  completed with finite metrics, four-environment coverage, epoch-2/5 checkpoints, clean `818fc8a`
  provenance, fatal-clear logs, `selection_split=ood_val`, and null heldout fields. Four auxiliary
  releases launched the E4 dense control and E16 learned/frozen/dense triplet; the top-2 releases
  launched the E4 top-2 pair. Six additional noise/image/E4-linear/E16-linear arms passed exact
  model construction with unique run IDs and restored ready depth to 12.
- **What we learned:** load-balance strength does not explain the sparse effect. OOD validation is
  `2.791/2.801/3.034/2.933%` at auxiliary weights `0.0001/0/0.05/0.1`. Zero balance lowers route
  entropy to `0.814` while keeping all eight experts active, yet route reliance falls to `0.00061`
  and accuracy declines. Thus uneven specialization alone is not useful routing. E8 top-2 learned
  scores `3.197%` versus `3.004%` frozen: a small `+0.193` OOD and `+0.239` ID-point difference,
  zero worst-environment difference, and route reliance only `0.00365`. It is a short family lead,
  not evidence that adaptive routing causes the improvement.
- **Correctness/trust:** these are exploratory seed-0 comparisons inside a multiple-search program.
  E8 top-2 learned/frozen share `38,950,261` total and `4,732,418` active FFN parameters. Auxiliary
  arms share the exact E8 top-1 class. OOD test remains sealed. No fresh seeds or long-horizon runs
  are licensed by these small results; the largest threat is multiplicity plus causal routing
  dependence remaining below `0.01`.
- **Traceable artifacts:** `analysis/fast_conflict_router_dynamics_epoch5_wave2_validation.json`,
  `analysis/fast_conflict_router_dynamics_epoch5_wave3_top2_validation.json`, the two queue-extension
  validations, the updated ready registry, and steward ledgers. HF remains quota-blocked and the
  manuscript is unchanged.
- **Next automatic action:** validate the E4 learned/frozen/dense and E4 top-2 learned/frozen
  families at their first completed checkpoint, refill from the 12 exact-built matched controls,
  and require a learned-versus-frozen difference with route reliance above `0.01` before spending
  beyond the short screen.

## 2026-08-04 04:18 EDT — E4 and routing-granularity rows close; matched mechanism controls refill

- **Where we are:** four additional epoch-5 rows are strict-valid. All ten schedulable H100
  assignments remain occupied after immediate refill: 2887 runs E4 dense and E16 learned; 2875
  now runs E16 top-2 learned/frozen; 2874 runs the frozen-noise and frozen-image controls; 2862
  runs E16 frozen/dense; and 2859 runs E4 top-2 learned/frozen. Zero usable H100s are idle. Four
  predeclared E4/E16 linear-top-2 controls restore the ready queue to 12; backlog remains 24.
- **What moved:** E4 learned/frozen, temperature 0.30, and image routing completed with finite
  metrics, four environments/9,854 samples, epoch-2/5 checkpoints, clean `818fc8a` provenance,
  fatal-clear logs, OOD-validation selection, and null heldout fields. An initial refill command
  failed before training because it used a relative config path; the one allowed narrow retry used
  the preflighted absolute config path and all four workers loaded cleanly with no excluded row.
- **What we learned:** E4 learned/frozen score `2.496/2.405%` OOD validation,
  `3.484/3.472%` ID, and `0.853/0.690%` worst environment. Learned-minus-frozen is only
  `+0.091/+0.012/+0.163` points, while learned route reliance (`0.00173`) is lower than frozen
  (`0.00315`). Temperature 0.30 and image routing score `2.628/2.659%` OOD and both trail the
  canonical E8 learned reference on mean, ID, and tail. Reducing expert count or making one image-
  level assignment does not reveal useful adaptive routing.
- **Correctness/trust:** all four rows are exploratory seed 0 after multiple searches. E4
  learned/frozen share `29,494,645` total and `2,366,210` active FFN parameters; the dense result
  is still finalizing and no sparse-versus-dense E4 claim is made here. OOD test remains sealed.
  The largest threat is that tiny early differences are optimization noise; the matched frozen-
  noise/image and top-2 pairs are the sharpest causal checks.
- **Traceable artifacts:**
  `analysis/fast_conflict_router_dynamics_epoch5_wave4_validation.json`, the updated ready registry,
  and steward ledgers. HF remains quota-blocked and the manuscript is unchanged.
- **Next automatic action:** validate E4 dense, E16 learned/frozen/dense, and E4 top-2 immediately
  when their result writers finish; refill each release from the 12-arm paired linear queue. Do not
  spend past epoch 5 unless a learned/frozen pair improves both decision metrics and raises route
  reliance above `0.01`.

## 2026-08-04 04:28 EDT — E4 benefit is mostly fixed partitioning; E16 favors dense

- **Where we are:** the complete E4 and E16 learned/frozen/dense matrices are strict-valid at
  epoch 5. All ten schedulable H100s are again assigned: 2887 runs E4 linear learned/frozen; 2875
  E16 top-2 learned/frozen; 2874 frozen-noise/image controls; 2862 E8 linear learned/frozen; and
  2859 E4 top-2 learned/frozen. Zero usable H100s are idle. Four E4/E16 symmetry-noise pairs restore
  the ready queue to 12; backlog remains 24.
- **What moved:** E4 dense and E16 learned/frozen/dense finalized with finite metrics, four
  environments/9,854 samples, epoch-2/5 checkpoints, clean `818fc8a`, fatal-clear logs,
  OOD-validation selection, and null heldout fields. Their four released GPUs were immediately
  refilled with paired linear-router screens.
- **What we learned:** E4 learned/frozen/dense OOD is `2.496/2.405/2.080%`. Learned exceeds exact-
  total dense by `+0.416` OOD points, but frozen already contributes `+0.325`; learned adds only
  `+0.091` over frozen and has lower route reliance. E16 learned/frozen/dense OOD is
  `2.760/2.770/3.024%`: learned is slightly below frozen and `0.264` points below dense, with route
  reliance only `0.00071`. The promising part is a small E4 sparse-partition regularization effect,
  not evidence that the router is adapting usefully.
- **Correctness/trust:** these are exact-total comparisons within E4 and E16 only, exploratory
  seed 0 after multiple searches. E4 sparse/dense totals are `29,494,645/29,493,881`; E16 are
  `57,861,493/57,860,753`. OOD test remains sealed. The largest threat is early-horizon noise and
  multiplicity; no epoch-10 continuation is licensed.
- **Traceable artifacts:**
  `analysis/fast_conflict_router_dynamics_epoch5_wave5_expert_count_validation.json`, the updated
  ready registry, and steward ledgers. HF remains quota-blocked and the manuscript is unchanged.
- **Next automatic action:** validate the E4 top-2 and matched frozen-noise/image controls at
  completion, then the linear pairs. Continue only a configuration that establishes both a useful
  learned-over-frozen effect and route dependence above `0.01` without losing tail performance.

## 2026-08-04 05:20 EDT — matched controls weaken the routing story; six configuration mismatches repaired transparently

- **Where we are:** four new epoch-5 rows are strict-valid. Ten short workers occupy all ten usable
  H100s: 2887 runs corrected E4 linear learned/frozen; 2875 corrected E16 top-2 learned/frozen;
  2874 E8 linear top-2 learned/frozen; 2862 corrected E8 linear top-1 learned/frozen; and 2859 E16
  linear top-1 learned/frozen. Zero usable H100s are idle. Four exact-built E2/E32 linear controls
  restore the runnable queue to 12; backlog remains 24. tester6/2899 remains scheduler-Pending.
- **What moved:** E4 top-2 learned/frozen and the frozen noise/image controls finalized with finite
  metrics, four environments/9,854 samples, epoch-2/5 checkpoints, clean `818fc8a`, fatal-clear
  logs, OOD-validation selection, and null heldout fields. Result metadata exposed that six other
  completed attempts had silently retained E8/cosine defaults because repeated `--override` flags
  were parsed incorrectly. Those six are preserved and explicitly excluded from their intended
  E16/E4/linear claims. One class-wide narrow repair relaunched them with a single override list;
  four independent linear pairs filled the other releases. All ten repaired/new identities were
  verified from their live W&B names and startup logs.
- **What we learned:** E4 top-2 learned/frozen OOD is `2.496/2.679%`; learned is `-0.183` OOD,
  `-0.131` ID, and `-0.203` worst-environment points below frozen, with lower route reliance.
  Noise-0.001 learned/frozen is `3.258/3.065%` OOD: only `+0.193` mean and `+0.066` ID points,
  while worst environment is `-0.041` and the reliance gap is only `+0.00041`. Image learned/frozen
  is `2.659/2.517%`: `+0.142/+0.039/+0.162` OOD/ID/worst points, but learned reliance is lower.
  These controls make fixed partitioning or ordinary optimization regularization more plausible
  than useful adaptive conditional computation at this short horizon.
- **Correctness/trust:** all accepted rows are exploratory seed 0 after multiple searches. The six
  mismatched attempts are never pooled or silently discarded; each has an intended identity,
  observed identity, exclusion consumer, and corrected r1 rerun in the validation artifact. The
  repaired launch uses the exact live execution tree at clean `818fc8a`. OOD test remains sealed.
  The largest threat is early-horizon/multiplicity noise plus route reliance staying below `0.01`.
- **Traceable artifacts:**
  `analysis/fast_conflict_router_dynamics_epoch5_wave6_controls_and_config_exclusions.json`, the
  updated ready registry, and steward ledgers. W&B is synced; HF remains quota-blocked and the
  manuscript is unchanged.
- **Next automatic action:** strict-validate the five corrected/new linear or top-2 pairs at their
  first completed checkpoint, refill each release from the 12-arm queue, and promote nothing unless
  learned routing gives a materially larger paired mean-plus-tail advantage with reliance above
  `0.01` and noncollapsed expert use.
