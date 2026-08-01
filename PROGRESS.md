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
