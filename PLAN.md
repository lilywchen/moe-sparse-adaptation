# RxRx1 sparse-adaptation kill test

## Authorized protocol revision — fast mechanistic discovery (2026-08-03)

The user has explicitly deprioritized additional broad 60--90 epoch exploration. Existing healthy
jobs that are already close to a declared checkpoint may finish because their marginal cost is
small, but a stalled or failed long-horizon arm must not be restarted merely to complete a matrix.
No new seed-0 recipe receives a 60- or 90-epoch budget until it first shows a short-horizon causal
signature that conditional computation is doing useful work.

The new discovery funnel is:

1. **Zero/short-training localization.** Use saved original Cell-DINO checkpoints to measure
   experiment-stratified gradient cosine, conflict rate, and norm at all 12 transformer FFNs.
   Training experiments only are used for gradients. Repeat enough balanced minibatches to
   identify a reproducible conflict peak and a low-conflict placebo layer.
2. **Two-epoch mechanism smoke.** At the conflict peak, compare function-preserving learned MoE,
   frozen-router MoE, and equal-budget dense controls. Cross one sparse FFN against a two-FFN
   conflict-localized intervention, and test staged router/expert unfreezing. Record route
   randomization, expert usage, entropy, augmentation stability, and pre/post gradient conflict.
3. **Five-epoch discriminator.** Continue only arms with valid, noncollapsed routing and either
   early paired OOD/tail movement or a strong causal mechanism signature. Use exact paired data
   order and controls; do not select on training loss.
4. **Ten-epoch confirmation screen.** A new recipe may exceed ten epochs only if it satisfies at
   least one accuracy route and the mechanism route: (a) at least +2 OOD-validation points versus
   its proper dense comparator with no worst-experiment loss, or a predeclared tail improvement;
   and (b) at least 1 absolute point route reliance, at least two materially used experts, and a
   reproducible reduction in cross-experiment gradient conflict. ID loss must not exceed 2 points.
5. **Long training is earned, not default.** Only a locked paired survivor receives epoch 30. Epoch
   60 and fresh seeds require persistence at 30; epoch 90 remains final adjudication only.

This revision does not invalidate completed long-run evidence. It changes prospective allocation:
the search now prioritizes information per GPU-hour and directly tests why one sparse FFN should
help. OOD test remains sealed, and all short-screen winners remain exploratory.

## Authorized protocol revision — Cell-DINO factorial60 exploration (2026-08-01)

The completed `hypothesis90` campaign remains a decision-grade negative for its exact canonical
recipe: seed-0 token top-1 MoE improves OOD validation by only 1.715 points over the exact-total-
parameter-matched dense-wide comparator and does not improve the worst experiment. It is not
relabelled. The user has now explicitly authorized a new, broader exploratory question and asked
that all available H100 capacity be used:

**Was the modest canonical result caused by choosing the wrong sparse placement, routing
granularity, router geometry, or environment pressure, and is any combination strong enough to
survive a locked paired confirmation?**

The new `factorial60` campaign revives the original theory-driven factorial on the now-qualified
native-CP5 Cell-DINO substrate. At seed 0 it runs:

- 36 MoE arms: 3 placements (early/middle/late) x 2 routing units (image/token) x 2 router
  geometries (linear/cosine) x 3 pressures (global canonical, within-experiment route balancing,
  experiment-adversarial output invariance);
- 6 exact-total-parameter-matched dense-wide controls: each placement under canonical and output
  pressure. Canonical dense controls serve both canonical and route-balanced MoE at the same
  placement; output-pressure MoE is paired only with output-pressure dense-wide;
- 1 original-width Cell-DINO reference.

All 43 arms use the same native-CP5 pixels, seed/data order, AdamW LR `1e-4`, weight decay `0.05`,
batch 64, drop path `0.1`, uniform layer LR, five warmup epochs, 8 experts, top-1 routing, and
official OOD-validation-only selection. They run for 60 epochs with exact train/ID/OOD-validation/
worst-experiment milestones at 10, 30, and 60. Every arm saves its final epoch-60 checkpoint;
original and the six dense comparators also save epochs 10 and 30 so all sparse winners retain a
reloadable paired anchor. Runs use a fresh W&B group and an isolated Hugging Face prefix. Upload is
deferred until strict result/checkpoint validation so network transfer never holds an H100 idle.

This is an exploratory factorial, so the number of searched cells must remain visible. Cell-level
rankings are provisional. A candidate advances only if, against its exact paired dense control at
the same placement and pressure, it improves OOD validation by at least 5 absolute points while
losing no more than 2 ID points; a smaller mean gain may advance only if a predeclared consistent
worst-experiment improvement is present. Advancing freezes the full configuration and launches
paired seeds 1 and 2; no further tuning may be performed on those confirmation seeds. OOD test
remains sealed until the separately frozen confirmatory stage.

The 10/30/60 milestones are also operational pruning gates. A scientifically dominated arm may be
stopped after a complete validated milestone only when its own predeclared delayed-emergence
alternative is false; released GPUs are immediately refilled from the remaining factorial queue.
Failures and pruned cells remain in the denominator and are never silently excluded. This revision
supersedes earlier statements that no broad Cell-DINO sweep was licensed, but does not change or
erase any earlier negative result.

## Authorized protocol revision — substrate-strength signal ladder (2026-08-01)

The completed ten-epoch kill campaign remains a valid negative result for its exact recipe; it is
not retroactively reinterpreted. The user has authorized a new bounded question before deciding
whether Cell-DINO is strong enough for a definitive architecture test:

**Can benchmark-strength supervised adaptation bring native-CP5 Cell-DINO close enough to the
completed local RxRx1 reference that a new sparse-capacity contrast has adequate power?**

The local canonical WILDS ResNet run completed 90 epochs and peaks at 19.5149% OOD-validation
accuracy at epoch 82, with 35.6249% ID accuracy and 99.9877% train accuracy. This is the local
positive control; the external leaderboard remains context rather than a selection split.

The user rejected a ten-way optimizer screen because it would spend the new parallel capacity on
knobs rather than explanations. The authorized batch is instead exactly ten seed-0, 90-epoch
native-CP5 runs from one tested tree. Every run uses AdamW LR `1e-4`, weight decay `0.05`, uniform
layer LR, batch 64, drop path `0.1`, five warmup epochs, the same data order and preprocessing, and
OOD-validation-only selection. Metrics are recorded at epochs 10, 30, 60, and 90; the original
anchor alone saves model checkpoints at all four milestones. Thus epoch dependence is read from
one trajectory rather than from duplicate budget arms.

The ten independent hypotheses are:

1. original Cell-DINO full fine-tuning (anchor);
2. exact-total-parameter-matched dense-wide capacity;
3. canonical global token top-1 MoE;
4. global image top-1 MoE (routing granularity);
5. token top-1 MoE with within-experiment load balancing (batch specialization pressure);
6. global token top-2 MoE (router starvation versus active-compute diagnostic);
7. frozen Cell-DINO plus linear head at LR `1e-3` (representation separability);
8. classifier plus last four transformer blocks (catastrophic forgetting/optimization depth);
9. full fine-tuning with an experiment-adversarial feature objective (explicit invariance);
10. full fine-tuning with equal per-experiment minibatch loss weighting (environment imbalance).

The 10/30/60/90 learning curves first decide whether the original result was simply undertrained.
At epoch 90, sparse efficacy is judged against dense-wide, not original width: a replicated campaign
is licensed only if a MoE arm exceeds dense-wide OOD validation by at least 5 absolute points while
losing at most 2 ID points. Top-2 is explicitly not an active-compute-matched claim. Frozen/partial,
adversarial, and environment-balanced arms are diagnostic interventions; they identify the failure
regime and may motivate a separately frozen confirmatory contrast, but cannot be pooled as MoE
replicates. No additional optimizer sweep is licensed by this batch.

The already-tested native-six Channel-Adaptive DINO pair remains an independent instrument arm,
licensed immediately after its distinct ViT-L/16 checkpoint passes integrity and load smoke. It
must not be simulated with the Cell-DINO ViT-S/8 checkpoint. OOD test remains sealed throughout.

## One scientific question

**Can sparse conditional computation make a microscopy-pretrained model substantially more robust
to unseen experimental batches by separating acquisition environments whose training gradients
interfere in a dense model?**

RxRx1 is the sole primary setting. It is not a convenience dataset: its 1,139 perturbations are
repeated across 51 experiments, and the official WILDS split separates 33 training, 4 validation,
and 14 test experiments. The label is stratified across experiments, so the benchmark directly
asks whether morphology transfers across batch rather than whether a batch shortcut predicts the
label. Camelyon17 is archived and receives no new compute; completed results may appear only as a
boundary-condition appendix.

## The problem must be diagnosed before it is treated

Low OOD accuracy has three technically distinct causes:

| Failure | Observable signature | Meaning for this paper |
|---|---|---|
| Representation/optimization failure | train and seen-experiment accuracy are low | substrate cannot yet expose perturbation signal; MoE comparison is invalid |
| Ordinary generalization failure | train high, seen-experiment accuracy low | task learning/regularization problem precedes batch robustness |
| Batch-transfer failure | seen-experiment accuracy is credible but OOD-val drops sharply | valid regime for testing conditional capacity |

Stage 0 therefore records training accuracy, ID-test accuracy, OOD-validation accuracy, the
ID-to-OOD gap, and per-experiment accuracy. A frozen-backbone linear probe asks whether the
pretrained representation already contains perturbation signal; two full-fine-tuning arms ask
whether supervised adaptation can recover it. Experiment and perturbation decodability are
diagnostics only and cannot substitute for predictive competence.

Natural-image DINOv2 is permanently excluded from RxRx1: its complete bounded rescue peaked at
5.50% OOD-val and 10.53% ID accuracy, only 35.7% of the canonical ResNet control's 15.42% OOD-val.
The first Cell-DINO instrument is also excluded: even after correcting the published downstream
pooling, the WILDS three-channel composite mapped as
`[DNA=nuclei, ER=ER, RNA=0, AGP=actin, Mito=0]` reaches only 17.76% train, 11.16% ID, and 5.93%
OOD-validation accuracy. It fails the frozen competence boundary and cannot license an MoE test.

The remaining bounded instrument comparison uses the official six grayscale RxRx1 acquisitions:

1. **Cell-DINO CP ViT-S/8 with a biologically matched fixed map:**
   `[DNA=w1, ER=w2, RNA=w4, AGP=mean(w3,w6), Mito=w5]`. RxRx1 records actin and Golgi separately;
   Cell Painting acquires both in AGP, so their fixed mean is formed before per-channel
   standardization. No learned adapter is introduced.
2. **Meta Channel-Adaptive DINO ViT-L/16 on native `w1..w6`:** the released Bag-of-Channels model
   consumes all six channels directly, avoiding the mapping assumption.

These are instrument diagnostics, not a backbone leaderboard. Meta authorship is not evidence of
fitness by itself; competence on train, ID, and OOD validation decides which substrate can support
the sparse-capacity question.

## Technical hypotheses

### H1 — Cross-batch gradient interference

For the same perturbation objective, gradients contributed by different training experiments are
less aligned at the converted FFN than gradients sampled within an experiment. A single dense FFN
must average these conflicting updates. Top-1 experts can reduce destructive averaging by giving
compatible examples partially separate update paths.

**Measurement:** experiment-stratified gradients at the selected FFN, pairwise cosine similarity,
gradient norm, and conflict rate (`cosine < 0`). After training, compare conflict before routing,
within each learned expert, and under randomized routes.

**Falsifier:** gradients are already aligned, or MoE does not reduce within-path conflict despite
an apparent accuracy gain.

### H2 — Reusable conditional structure, not experiment memorization

Useful routes should capture recurring acquisition/morphology modes shared across experiments.
They may be batch-decodable—batch is genuinely visible—but must not collapse into one expert per
training experiment. The same routing rule must remain useful on held-out experiments.

**Measurement:** expert usage by experiment and perturbation, routing entropy, dead experts,
cross-validated experiment/label decodability, route stability across augmentations, and held-out
experiment usage. Always compare with a frozen router and post-training randomized routes.

**Falsifier:** routes merely identify training experiments, held-out experiments collapse to a
small set of experts, or randomizing routes leaves performance unchanged.

### H3 — Conditional capacity improves batch robustness, not merely fit

Relative to the total-parameter-matched dense-wide FFN, MoE should improve OOD-validation and
worst-experiment accuracy, reduce the ID-to-OOD degradation gap, and preserve ID accuracy. The
original-width model is reported only as an accuracy/compute reference.

**Primary estimand:**

`conditional_gain = OOD-val accuracy(MoE) - OOD-val accuracy(dense-wide)`.

The target is a reproducible 10–15 absolute percentage-point gain. A smaller 3–5 point paired gain
can justify full replication only if it is consistent by experiment and accompanied by the
predeclared mechanism. A one-seed 1–2 point difference is not a result.

## Minimal experiment sequence

### Stage 0A — substrate competence and failure decomposition

For each of the two native-channel instruments, run exactly one paired seed-0 diagnostic on OOD
validation only:

1. frozen backbone + linear head, 5 epochs, LR `1e-3`;
2. full fine-tuning, 10 epochs, LR `1e-4`, uniform layer LR.

Both use official-style RxRx1 geometry, per-image/per-channel standardization, the same WILDS
train/ID/OOD-validation split, no photometric augmentation, and the published
class-token-plus-mean-patch-token representation. Cell-DINO uses 128×128 input and the frozen CP5
map above; Channel-Adaptive DINO uses 224×224 and native six-channel input. Do not add learning-rate
or preprocessing arms after looking at results.

Competence requires a clearly learned perturbation task: the full-fine-tuned model must improve
materially over its frozen probe, show nontrivial ID accuracy, and reach the canonical ResNet sanity
range on OOD validation. If neither does, stop: this is not yet a conditional-capacity experiment.
If both qualify, choose the stronger OOD-validation instrument, breaking ties with worst-experiment
accuracy; this choice precedes and cannot be revisited by the MoE comparison.

### Stage 0B — the kill comparison

Using the one frozen recipe and seed 0, run:

1. original-width Cell-DINO (reference);
2. one middle-block dense-wide model;
3. one middle-block learned MoE: 8 experts, top-1 token routing, cosine router, standard global
   load balancing.

No placement, router, loss, or expert-count grid is allowed before this comparison. Dense-wide and
MoE must have the same total parameter count within 0.1%, same initialization function, data order,
optimizer, schedule, training steps, seed, classifier, and validation calls. The MoE has lower
active FFN compute by design; report total parameters, active parameters, throughput, and memory.

### Stage 0C — replication trigger

If seed-0 MoE improves OOD-val by at least 5 points without losing more than 2 ID points, run paired
dense-wide/MoE seeds 1 and 2 immediately. If the gain reaches 10 points, treat it as a high-priority
signal but still require those replications. If it is below 5 points, do not launch a grid; run only
the already-predeclared frozen/random-route diagnosis, then decide whether the premise survives.

### Stage 1 — one-dataset generalization after a real effect

Only after a replicated effect:

- construct several experiment-held-out folds from official training experiments, stratified by
  cell type, without touching OOD test;
- report paired per-experiment differences and cluster-bootstrap confidence intervals;
- run a fixed-sample/step nested 4/8/16/32-training-experiment curve to test whether conditional
  gain grows with batch heterogeneity;
- measure H1/H2 mechanisms on the successful model and the matched dense control.

This turns one dataset into many acquisition replications and makes a negative result informative:
if conditional capacity fails across favorable folds and increasing heterogeneity while gradients
conflict, then sparse routing does not solve the interference it was designed to address.

### Stage 2 — confirmatory test

Freeze the exact architecture, recipe, seeds, fold interpretation, and analysis code. Only then run
fresh confirmatory seeds and evaluate the official OOD test once. OOD test cannot influence any
configuration choice.

## Correctness gates

- Every JSON must be parseable, finite, exact-config/seed/run-ID matched, clean-provenance, and
  record `selection_split=ood_val` and `test_evaluated=false` before Stage 2.
- Checkpoint filename, SHA-256, public Cell-DINO code commit, channel mapping, and backbone source
  are written into each result.
- Dense-wide and MoE total parameters differ by less than 0.1%; no padding parameters.
- Every reported MoE result has its paired dense-wide result at the same seed.
- Report training, ID, OOD-val, worst experiment, degradation gap, parameter counts, throughput,
  exclusions, and uncertainty. Never silently drop a failed experiment or run.
