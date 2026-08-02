# Progress ledger

Last verified: 2026-08-01 12:17 EDT on SciServer and GitHub. The paper has not been changed for the
new campaign because no new performance milestone is valid yet.

Research-state synchronization: GitHub commit `75c2e85`, containing the validated negative kill
gate and manuscript table, was pulled into the linked Overleaf project on 2026-08-01;
`paper/main.tex` compiled successfully to five pages with 0 errors and one pre-existing warning.
Overleaf then reported no newer GitHub commit since the merge.

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
