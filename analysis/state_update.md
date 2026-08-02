# Living scientific state

Last verified: 2026-08-02 05:16 EDT

## Scientific question

When a pretrained vision transformer is fully adapted to scientific images collected across
acquisition environments, does sparse conditional capacity improve held-out-environment
generalization over a total-parameter-matched dense model, and which routing choices explain the
gain or failure?

## Where the project is

The user approved Cell-DINO Cell Painting ViT-S/8 as the replacement RxRx1 substrate. The exact
RxRx1-only kill-test implementation is on GitHub commit `db9ebdb` and in a tree-identical isolated
SciServer checkout (tree `c537d33ef4db3dc2206c0922a0159b4b19095adc`). The official DINOv2
source is pinned at `7764ea0`, and the complete SciServer test suite passes 80/80 tests.

The checkpoint is now reconstructed and strictly load-validated at the frozen persistent path. Its
86,164,384 bytes hash to `37d20e9cd48b3d610b5de15a4ea4e7e060a593b8d8358e928d079dc7b03ee66a`,
matching the hash prefix in Meta's filename. A real-checkpoint model smoke produces `1x1139` logits
from `1x5x128x128` input through 12 transformer blocks.

The next scientific gate is not MoE yet. It is the three-way Cell-DINO failure decomposition:
frozen-backbone linear probing, full fine-tuning at `1e-4`, and full fine-tuning at `3e-4`, with
train, seen-environment, OOD-validation, and worst-experiment accuracy. The linear probe is strictly
valid, and both full-fine-tuning jobs are healthy on two H100s; none is queued. No new Camelyon17
work is licensed.

The study is in Stage 0. The frozen 36-cell factorial has not started. All six RxRx1 seed-0 shared-
HPO candidates have completed and passed strict validation under execution commit `26ad7fa`.
Three Camelyon17 candidates are valid, giving `9/12` formal results overall. All three remaining
Camelyon17 candidates are active; the last missing shard was dry-run and launched on an idle GPU.

The RxRx1 DINOv2 competence gate is resolved as a failure. All four bounded rescue diagnostics are
strictly valid, and the best reaches only 35.7% of the canonical control's best-so-far OOD-val
accuracy, below the predeclared 50% abandonment threshold. DINOv2 is excluded from RxRx1 MoE work.
The recommended replacement candidate is microscopy-pretrained Cell-DINO Cell Painting ViT-S/8;
selecting and qualifying it is now the only RxRx1 design decision. The OOD-test subset remains
unavailable.

## Decision-grade findings

No MoE-versus-dense performance finding is decision-grade yet. The decision-grade Stage-0 result is
that natural-image DINOv2 is not a competent RxRx1 substrate under the complete predeclared rescue
set and must not consume the factorial budget. There is not yet a valid basis for freezing an
RxRx1 replacement recipe, selecting a Camelyon17 recipe, comparing MoE with dense controls, or
making a mechanism claim.

The following protocol facts are verified:

- the formal Phase-A revalidation uses one tested code commit (`26ad7fa`);
- all Stage-0/1 selection remains on `ood_val`, with `test_evaluated=false` required;
- the earlier 12-result grid spans multiple commits and is diagnostic only;
- the RxRx1 canonical reproduction has an explicit validation-only split guard.

Six RxRx1 results have passed strict validation: parseable JSON, finite headline metrics,
exact run/config/filename identity, clean common commit `26ad7fa`, `selection_split=ood_val`,
`test_evaluated=false`, and equal training parameter count (21,628,800). Their metrics are:

| Learning rate | LLRD | OOD-val accuracy | seen-environment accuracy | worst-environment val | status |
|---:|---:|---:|---:|---:|---|
| 1e-4 | 0.70 | 0.01035 | 0.02896 | 0.00081 | valid |
| 1e-4 | 0.85 | 0.01482 | 0.03804 | 0.00203 | valid |
| 3e-5 | 0.70 | 0.00913 | 0.01820 | 0.00122 | valid |
| 3e-5 | 0.85 | 0.00995 | 0.02187 | 0.00081 | valid |
| 3e-4 | 0.70 | 0.01025 | 0.02467 | 0.00081 | valid |
| 3e-4 | 0.85 | 0.01340 | 0.04481 | 0.00081 | valid |

## Provisional diagnostics

The frozen Cell-DINO linear probe is strictly valid and reaches 5.83% training accuracy, 4.05%
seen-environment accuracy, 2.87% OOD-validation accuracy, and 0.97% worst-experiment validation
accuracy. This establishes that a frozen linear readout is weak. It does not distinguish a failed
Cell-DINO substrate from a representation that requires full adaptation, so the two full-fine-
tuning arms remain the decisive competence evidence.

The `1e-4` full-fine-tuning arm is also strictly valid and reaches 14.46% train, 8.59% seen-
environment, 4.84% OOD-validation, and 1.38% worst-experiment validation accuracy. The gain over
the linear probe supports an optimization/adaptation component, but the absolute seen and OOD
levels remain weak. This remains provisional until the `3e-4` arm finishes; if it does not improve
substantially, the issue is not merely that the Cell-DINO backbone was frozen.

A frozen non-parametric representation probe is now running. Exact cosine 1-NN and nearest-class-
centroid readouts answer a sharper question than another trained head. An ID-high/OOD-low 1-NN
pattern would diagnose experiment-local neighborhoods and therefore a genuine batch-transfer
failure in the pretrained geometry. A low/low pattern for both readouts would instead diagnose
weak perturbation encoding. Centroid improvement on OOD validation would indicate that cross-batch
class averaging removes nuisance structure without changing the backbone.

The pretraining-overlap audit found no documented RxRx1 exposure: the Cell-DINO paper describes a
five-dataset combined Cell Painting pretraining resource and explicitly leaves RxRx-series work as
a future direction. Domain-matched self-supervision is therefore an intended advantage, not known
benchmark-image leakage. This conclusion should be revised if Meta releases a more granular image
manifest contradicting the paper.

Two 90-epoch DINOv2 RxRx1 probes at learning rate `1e-4` produced:

| LLRD | OOD-val accuracy | seen-environment accuracy | status |
|---:|---:|---:|---|
| 0.70 | 0.0140 | 0.0945 | diagnostic only |
| 0.85 | 0.0177 | 0.1042 | diagnostic only |

Both files are parseable, use `selection_split=ood_val`, and have `test_evaluated=false`, but they
were produced from dirty commit `4795202`; they cannot be used for formal ranking. Longer training
therefore improved optimization relative to the short probes but did not establish a scientifically
credible RxRx1 adaptation result.

The canonical WILDS reproduction is healthy at epoch 22. At epoch 21 it reached 70.7% training-set
evaluation accuracy, 24.6% ID-test accuracy, and 13.4% OOD-validation accuracy. This is diagnostic,
not a contender in the MoE comparison, but it establishes that the current RxRx1 data, labels, and
split plumbing support substantial learning. It strongly shifts the diagnosis away from a broken
dataset and toward the DINOv2 adaptation substrate.

The rescue set is fixed relative to the best DINOv2 anchor `(lr=1e-4, LLRD=0.85)`:

| Run | Crop | LLRD | Normalization / geometry | Role |
|---|---|---:|---|---|
| anchor | random resized | 0.85 | ImageNet / flips | failed reference |
| `rxdiag_no_rrc` | resize only | 0.85 | ImageNet / flips | crop main effect |
| `rxdiag_uniform_lr` | random resized | 1.0 | ImageNet / flips | adaptation main effect |
| `rxdiag_no_rrc_uniform_lr` | resize only | 1.0 | ImageNet / flips | interaction |
| `rxdiag_wilds_uniform_lr` | resize only | 1.0 | per-image channel standardization / right-angle rotations + horizontal flip | Rx-native preprocessing |

All rescue arms are diagnostic-only and cannot enter the formal HPO ranking. The official-style
implementation passed 3 focused tests and 75/75 full-suite tests before launch.

All rescue results are now valid:

| Run | OOD-val accuracy | seen-environment accuracy | worst-environment val | Status |
|---|---:|---:|---:|---|
| `rxdiag_no_rrc` | 0.015425 | 0.045134 | 0.001218 | valid diagnostic |
| `rxdiag_uniform_lr` | 0.013294 | 0.036738 | 0.000812 | valid diagnostic |
| `rxdiag_no_rrc_uniform_lr` | 0.018571 | 0.052300 | 0.002029 | valid diagnostic |
| `rxdiag_wilds_uniform_lr` | 0.055003 | 0.105289 | 0.010146 | valid diagnostic |

Relative to the fixed anchor (0.014816 OOD-val, 0.038043 seen), removing crop gives only a small
gain, uniform layer learning rates alone are worse, and their interaction remains weak. Official-
style preprocessing is materially better, establishing a real preprocessing mismatch, but its
0.055003 OOD-val accuracy is only 35.7% of the canonical control's best-so-far 0.154151. This
triggers the frozen below-50% abandonment rule on OOD validation alone.

## Current scientific interpretation

### Cell-DINO frozen geometry and current adaptation regime

**Qualification is reopened under one implementation-faithful correction.** The initial runs used
only the CLS token, whereas the Cell-DINO paper's downstream representation concatenates the
normalized CLS token with the mean of normalized last-block patch tokens. This is not post-hoc
hyperparameter tuning: the data, optimizer, schedule, seeds, learning rates, and selection policy
remain fixed while the representation changes from 384 to 768 dimensions. The CLS-only results
remain valid diagnostics but are superseded for the final instrument decision.

The exact three-arm competence set is now running with official pooling. If the correction raises
train and seen-environment accuracy substantially, it isolates the readout mismatch as a real defect.
If it does not, the strongest remaining explanation is the input mismatch: the WILDS view supplies
three channels to a five-channel checkpoint, leaving two pretrained stain slots permanently zero.
That outcome would support using the original multichannel RxRx1 data or a channel-compatible
microscopy substrate, not further unbounded tuning.

The official-pooling frozen linear probe is now strictly valid. It reaches 7.52% train, 4.61% seen,
and 3.03% OOD-validation accuracy, versus 5.83%, 4.05%, and 2.87% for CLS alone. The additional patch
summary therefore exposes slightly more linearly accessible signal but does not materially improve
unseen-batch transfer; worst-experiment OOD validation actually changes from 0.97% to 0.85%. This is
provisional diagnostic evidence only. The two full-fine-tuning arms must complete before deciding
whether official pooling meaningfully changes dense competence.

The full Cell-DINO competence set and a stricter out-of-box representation probe are now complete.
With every Cell-DINO weight frozen, exact cosine 1-NN reaches 2.52% on ID-test and 1.23% on OOD
validation; nearest-class-centroid is lower at 1.71% and 0.76%. Thus the pretrained embedding does
not already organize RxRx1 perturbation classes in a way that either local retrieval or simple
cross-batch class averaging can recover. In particular, centroid aggregation does not rescue OOD
performance, so the result does not support the narrow hypothesis that only local neighbours are
batch-confounded while global class geometry is already correct.

Full fine-tuning helps, but the best bounded arm (`lr=1e-4`) reaches only 14.46% train, 8.59% seen,
and 4.84% OOD-validation accuracy; `lr=3e-4` is worse at 8.93%, 6.07%, and 3.74%. Because train and
seen accuracy are both low, this is a representation/optimization competence failure under the
current recipe. It is not yet evidence that the model specifically confuses unseen batches, and it
cannot adjudicate the MoE hypothesis.

The main competing explanations are now (a) an unverified mismatch in Cell-DINO channel order or
normalization, (b) insufficient or inappropriate supervised adaptation relative to the released
Cell-DINO evaluation protocol, and (c) a genuine mismatch between the checkpoint's learned Cell
Painting geometry and RxRx1 perturbation discrimination. The first two must be audited in a bounded,
implementation-faithful diagnosis before concluding (c). No additional MoE architecture sweep is
scientifically useful until train/seen competence is established.

The complete clean-commit RxRx1 grid establishes that the current DINOv2 recipe is not merely
poorly tuned within the tested LR/LLRD screen: its best OOD-validation accuracy is 1.48% and its best
seen-environment accuracy is 4.48%, while the canonical control already reaches 13.4% and 24.6%
respectively. This is still not evidence against MoE or batch-effect modeling; every weak model is
the original dense DINOv2 backbone. The evidence now says that the experimental instrument must be
repaired or replaced before the architectural question can be tested.

Competing explanations remain live:

1. **Preprocessing mismatch:** the DINOv2 pipeline removes or distorts signal that the canonical
   RxRx1 pipeline preserves.
2. **Optimization mismatch:** the pretrained transformer needs a different schedule or adaptation
   budget than the current bounded probes.
3. **Representation mismatch:** natural-image pretraining may genuinely transfer poorly to this
   microscopy task.
4. **Residual implementation mismatch:** less likely now, but still possible if the two pipelines
   differ in a way beyond the intended model/transform/optimization choices.

The canonical curve has resolved the main data-pipeline ambiguity: RxRx1 is learnable here. The
bounded rescue set now distinguishes destructive cropping, suppressed early-layer adaptation,
their interaction, and a domain-specific preprocessing mismatch. DINOv2 is retained automatically
only above 80% of the canonical control on both OOD-val and ID-test accuracy, abandoned
automatically below 50% on either, and requires one explicit decision in between. This prevents
the strength of DINOv2's reputation from turning into unlimited tuning.

The complete rescue set resolves the leading alternatives. Cropping and LLRD were not the primary
failure. Official RxRx1 normalization and augmentation semantics explain a substantial fraction of
the gap, but not enough to make natural-image DINOv2 a competent experimental instrument. The most
plausible remaining explanation is representation-domain mismatch: the pretrained features are
strong for natural images yet poorly aligned with subtle fluorescence phenotypes. This is a
substrate result, not evidence for or against MoE.

## Implications for experimental design

- Preserve the formal RxRx1 DINOv2 ranking as diagnostic history, but do not replicate or use it.
- Keep the factorial unchanged while replacing only the failed RxRx1 substrate. Recompute exact
  dense/MoE parameter matching for the replacement architecture and re-run the dense competence
  gate before any RxRx1 router calibration.
- Do not change factorial factors, levels, losses, or selection rules in response to this diagnostic.
- Camelyon17 may advance independently when its six valid seed-0 candidates complete; it need not
  wait for the RxRx1 diagnosis.
- Treat routing, decodability, and embedding analyses as mechanism evidence only after the matched
  predictive comparison is valid.

## Next decisions

1. Complete and strictly validate the three official-pooling Cell-DINO competence arms, then apply
   the instrument threshold to the corrected representation.
2. If dense competence is established, freeze one recipe and run the seed-0 original versus exact
   total-parameter-matched dense-wide versus canonical-MoE kill contrast.
3. Replicate only if MoE gains at least 5 absolute OOD-validation points while losing no more than
   2 ID points; prioritize the 10--15 point effect target.

## Current native-channel interpretation

The official-pooling rerun resolves the zero-filled WILDS composite instrument as a failure. Its
best full-fine-tuned arm learns more than the frozen probe but remains weak at 17.76% train, 11.16%
seen, and 5.93% OOD validation. This does **not** yet say that RxRx1 is too OOD for Cell-DINO. The
failed input omits RNA and mitochondria entirely and represents only actin in a pretrained AGP slot
that normally combines actin and Golgi. Representation/optimization failure remains the correct
classification; batch confusion is not established.

The native six-channel experiment cleanly separates two competing explanations:

1. **Missing-stain/interface mismatch.** If biologically mapped native Cell-DINO sharply raises
   train and ID accuracy, the RGB/WILDS proxy—not the checkpoint—was the dominant failure.
2. **Need for channel-set adaptation.** If native CP5 remains weak but Channel-Adaptive DINO is
   competent, forcing six acquisitions into a fixed five-stain interface is the dominant problem.
3. **Task/substrate mismatch.** If both remain weak, neither microscopy pretraining scheme exposes
   the 1,139 perturbations under the bounded adaptation budget; an MoE experiment would still be
   invalid, regardless of model reputation.

The mapping is biologically defensible rather than arbitrary: Hoechst→DNA, ConA→ER, Syto14→RNA,
MitoTracker→Mito, and Phalloidin+WGA→AGP. Averaging the two AGP components before per-channel
standardization is equivalent to summing up to scale and approximates their joint Cell-Painting
acquisition. Channel-Adaptive DINO is the necessary mapping-free control. The comparison is an
instrument qualification, not a claim that one backbone architecture is generally superior.

The highest-value next evidence is the paired frozen/full-fine-tuned result for each instrument.
A large train/ID recovery with a remaining OOD gap would finally enter the genuine batch-transfer
regime and license the matched dense-wide versus MoE kill contrast. Failure of both instruments is
also informative: it falsifies the current experimental instrument before expensive sparse-model
work and prevents a misleading negative MoE conclusion.

## Native preflight interpretation

The biologically mapped Cell-DINO interface is now executable with the released checkpoint. A real
six-channel sample maps to the expected five CP slots and produces finite 1,139-class logits with
the paper-specified pooled feature. This removes implementation incompatibility from the leading
explanations, but it is not yet evidence of representation quality: the competence result still
requires full selection-split coverage and the paired frozen/full-fine-tuning runs.

The incomplete first channel audit is explicitly non-scientific. Missing paths were present in the
official archive but had not yet been extracted, so no claim about stain availability, experiment
coverage, or model performance may consume that report. A skip-existing extraction is completing
the same immutable archive before the audit is rerun.

Channel-Adaptive DINO is valuable because its Bag-of-Channels encoder processes the six acquisitions
as a set of independently encoded channels rather than forcing them into a fixed five-slot stem. If
it succeeds where mapped Cell-DINO fails, the clean interpretation is an input-interface/channel-set
problem, not generic microscopy pretraining quality. If both succeed similarly, the fixed map was
sufficient and the smaller Cell-DINO substrate is preferable for the matched MoE kill test. If both
fail, the bounded adaptation regime is not a competent RxRx1 instrument and MoE remains untested.

The Channel-Adaptive comparison remains scientifically useful, but only through the official
Bag-of-Channels feature construction now covered by the tested adapter. It is deliberately an
instrument diagnosis rather than an extra tuning arm: the comparison asks whether preserving all
six acquisitions without a hand-designed CP5 map materially changes train, seen-environment, and
OOD-validation competence. A Channel-Adaptive win would implicate the fixed stain interface; a tie
would favor the smaller mapped Cell-DINO substrate; joint failure would reject both bounded
instruments. No performance conclusion follows from the adapter repair itself.

The post-extraction watcher changes turnaround time, not the scientific design. It can launch only
the already frozen native CP5 diagnostic pair after the complete selection-split audit passes; any
missing sample, dirty tree, duplicate process, missing checkpoint, or occupied GPU prevents launch.
There is still no native-channel performance evidence to interpret.

## Native CP5 competence launch interpretation

The complete audit removes incomplete extraction as a live explanation for the native result:
train, ID-test, and OOD-validation contain 91,078 samples with zero missing six-channel acquisitions.
The frozen probe and full fine-tune are now running from one tested tree and one fixed biological
mapping. This is still diagnostic evidence, not a model-selection or MoE result; epoch-0 loss only
shows that optimization is advancing.

The paired design answers a sharper question than another learning-rate sweep. A weak frozen probe
with a strong full fine-tune would mean the microscopy checkpoint contains useful but not linearly
separable structure that joint adaptation can recover. Jointly weak train and ID accuracy would
implicate the CP5 mapping/substrate or task interface rather than batch transfer. Credible train and
ID accuracy with a substantially lower OOD-validation and worst-experiment score would finally
establish the intended batch-transfer failure regime in which conditional capacity is scientifically
testable. No one of these interpretations is licensed until the corresponding result JSON passes
strict validation.

The most important remaining validity threat is instrument competence, not parameter fairness: the
MoE and dense-wide models have not been launched. Their exact total-parameter match, shared recipe,
paired seed, and ID-preservation checks remain frozen for the next gate. Six H100s are intentionally
idle because extra diagnostic arms cannot resolve the current decision, while Channel-Adaptive DINO
remains blocked on its distinct approved ViT-L/16 checkpoint.

## Native competence decision and active kill contrast

The native CP5 result changes the scientific diagnosis. Full fine-tuning reaches 39.00% train,
27.09% ID, and 12.29% OOD-validation accuracy, whereas the frozen probe reaches 10.62%, 7.64%,
and 4.42%. Adaptation therefore recovers substantial task signal, and the full model is in the
canonical WILDS sanity range. The remaining 14.80-point ID--OOD gap is now the primary phenomenon:
the model can recognize perturbations from seen experiments but transfers much less reliably to
four unseen experiments. This is decision-grade instrument qualification and establishes the
batch-transfer regime; it is not yet a sparse-model result.

The main live explanations are now sharper. Dense shared FFNs may suffer experiment-stratified
gradient interference; alternatively, the gap may be ordinary overfitting or an acquisition shift
that conditional routing cannot repair. The active seed-0 contrast distinguishes useful conditional
capacity from ordinary capacity by comparing MoE directly with an almost exactly total-parameter-
matched dense-wide model. A MoE improvement only over the smaller original model would not support
conditional specialization.

The worst held-out experiment remains near floor at 1.42%, so average OOD validation could conceal
strong experiment heterogeneity. A positive result must therefore preserve ID accuracy and improve
the per-experiment distribution, not merely the mean. Conversely, a null MoE-minus-dense-wide
contrast is informative: under a competent microscopy substrate and a controlled batch-transfer
gap, sparse conditional capacity alone would not solve the failure.

The three matched seed-0 jobs are active. No interpretation is licensed from their early losses.
The next decision is mechanical: validate all three JSONs, compute MoE minus dense-wide OOD-val and
the MoE ID difference, and replicate only for at least +5.0 OOD points with no more than -2.0 ID
points. OOD test remains sealed.

## Seed-0 kill decision: a small conditional recovery, not the target effect

The matched seed-0 contrast is complete and strictly valid. Canonical MoE reaches 11.61% OOD
validation versus 10.46% for the exact-total-parameter-matched dense-wide model, a +1.15-point
conditional gain. ID accuracy also rises by 1.98 points and every held-out experiment moves in the
same direction, so the MoE is not simply trading away seen-environment performance. This is the
most favorable reading of the result.

The decisive counterevidence is scale and reference ordering. The frozen gate required at least a
+5-point gain, while the smaller original Cell-DINO reaches 12.31% OOD validation and remains 0.70
points above MoE. Added dense width hurts; conditional routing recovers some of that harm but does
not surpass the original substrate. The 10--15-point effect that motivated rapid expansion is not
present in this canonical seed-0 contrast, so seeds 1/2 and the architecture grid are stopped.

Scientifically, H3 is not supported at the target scale: equal total capacity does not make MoE a
substantially more robust model here. H1 and H2 remain diagnostic possibilities only. The consistent
per-experiment direction could reflect modest protection from dense gradient interference, but it
could also reflect optimization differences or regularization induced by sparse activation. The
active frozen/random-route pair asks whether learned routing actually matters; it cannot turn a
failed predictive gate into a positive efficacy claim.

The most important remaining threat is single-seed uncertainty. The gate is nevertheless valid
because its stopping rule was frozen before the result and the observed gain is far below the
replication threshold. The correct conclusion is not “MoE never helps batch effects”; it is that
this canonical token-routed middle-FFN intervention does not justify a larger RxRx1 campaign under
the promised effect-size criterion. OOD test remains sealed.

The next decision is bounded: validate the learned/randomized-route and frozen-router diagnostics,
then decide whether the small dense-wide recovery has a coherent routing mechanism worth reporting
as a negative mechanistic study. No further architecture search is licensed.

## Provisional routing diagnosis and validity repair

The first bounded diagnosis does not yet license a mechanism conclusion. Its performance pattern is
scientifically suggestive: the learned router reaches 11.42% OOD validation, randomizing its routes
reduces accuracy by only 0.04 points, and freezing the router instead reaches 11.98%. If reproduced
cleanly, that combination would strongly disfavor H2: the learned assignments would carry little
causal predictive value, and learning the router would not outperform a fixed partition.

Two alternative explanations remain. Randomized rerouting could be a weak intervention if experts
never differentiated meaningfully, or the token router could encode acquisition/biology despite
predictions being insensitive to those assignments. The intended routing-MI audit distinguishes
these cases, but it failed because token assignments were paired with image-level labels without
repetition. Therefore the apparent low route reliance is provisional and the MI/usage evidence is
absent, not null.

The exact defect is repaired and the same two arms are running once from a clean, tested checkout.
The final interpretation requires aligned token-label counts, finite routing MI/entropy/usage,
clean provenance, and reproduction of the learned-versus-frozen/randomized performance ordering.
If learned routing remains nearly irrelevant, the clean negative story becomes sharper: conditional
capacity modestly regularizes a harmful dense widening, but the router does not discover reusable
batch structure. If route reliance or aligned MI changes materially, the competing routing
explanation remains live. Neither outcome can reopen the failed +5-point efficacy gate.

## Final bounded routing diagnosis

The clean retry resolves the mechanism ambiguity in the unfavorable direction for H2. The learned
router reaches 11.65% OOD validation; replacing its assignments with seeded random routes reduces
accuracy to 11.23%, a reliance of only 0.42 points. Freezing the router during training reaches
11.84%, 0.19 points above the learned router, and has 0.71-point randomized-route reliance. Both
models use all eight experts, but normalized routing entropy is 0.9999 for learned and 0.9975 for
frozen routing. Aligned route mutual information is very small: experiment/class MI is
0.0051/0.0016 for learned routing and 0.0098/0.0041 for frozen routing.

These measurements rule out the audit-failure explanation and disfavor reusable learned
specialization. The router neither concentrates into distinct experts nor carries appreciable
experiment or perturbation information on the reported audit scale, and learning it does not beat
a fixed partition. The nonzero randomized-route drops show that experts adapt weakly to their
partitions, but less than one accuracy point of dependence is not the proposed 10--15-point batch
robustness mechanism.

The clean scientific conclusion is narrower than “MoE cannot help microscopy.” In this competent
native-stain Cell-DINO/RxRx1 regime, the canonical middle-block top-1 token MoE recovers only 1.15
points relative to a parameter-matched dense widening, remains below the smaller original model,
and shows no evidence that a learned router discovered reusable batch structure. Sparse activation
may regularize an over-wide comparator or create weak random subspaces; single-seed variation also
remains possible. H1 is not established because the stopped campaign did not license a deeper
gradient-conflict study. The predeclared negative gate is complete and no further architecture,
mechanism, replication, or OOD-test experiment is licensed under that completed protocol.

## Authorized substrate-strength hypothesis campaign

The user has opened a new bounded question rather than retroactively changing the completed kill
gate. The strongest local positive control is now the completed canonical WILDS ResNet trajectory:
19.51% OOD validation at epoch 82, with 35.62% ID and 99.99% train accuracy. Native CP5 Cell-DINO at
ten epochs was meaningfully learned but below that reference. The live issue is therefore whether
Cell-DINO was under-adapted, whether the original sparse intervention was technically mismatched to
the batch failure, or whether the remaining gap is not addressable by conditional capacity.

The ten live arms are a hypothesis matrix, not a tuning sweep. A single 90-epoch original anchor
with 10/30/60/90 checkpoints tests duration without duplicating four training budgets. Dense-wide
and canonical MoE retest conditional versus shared capacity at sufficient duration. Image routing
tests whether experiment-level acquisition context is diluted by token routing; within-experiment
load balancing tests whether global balancing suppressed batch specialization; top-2 tests whether
hard top-1 assignment starved expert learning. Frozen and last-four-block adaptation test linear
separability versus catastrophic forgetting/optimization depth. Experiment-adversarial output
invariance and environment-balanced loss test two non-MoE explanations for the ID--OOD gap.

All ten workers have completed epoch 1 under one clean tested execution tree and one native CP5
data/split family. This is operational evidence only. No new predictive comparison is valid until a
milestone file passes strict identity, finiteness, coverage, provenance, and test-blindness checks.
The most important competing explanations remain: (1) ten epochs underfit all transformer-based
arms; (2) routing granularity or load-balance pressure prevented specialization; (3) shared
representation adaptation, not conditional FFN capacity, dominates; or (4) the observed gap is an
ordinary hard-domain generalization problem. The milestone trajectories were chosen because each
of these explanations predicts a different ordering, not merely a different scalar learning rate.

The campaign uses a fresh W&B group and will publish each completed folder to Hugging Face only
after strict validation. This improves traceability but does not make interim losses or W&B status
scientific evidence. The original versus dense-wide versus MoE epoch-90 comparison remains the
primary efficacy contrast; top-2 has different active compute, and frozen/partial/invariant/
environment-balanced arms remain diagnostic. OOD test is untouched.

The active 15-minute steward now encodes this exact interpretation and must treat completion as a
validation/publishing handoff rather than an invitation to add tuning arms.

## Hypothesis90 epoch-10 trajectory: no sparse-transfer signal yet

All ten epoch-10 milestone rows now pass the predeclared identity, finiteness, split, and
test-blindness checks. The original anchor leads OOD validation at 13.77%, followed by token top-2
MoE at 13.63% and matched dense-wide at 13.31%. Canonical token top-1 MoE reaches 12.90%, which is
0.42 percentage points below dense-wide; image top-1 and within-experiment-balanced routing are
also below dense-wide. Top-2 is 0.31 points above dense-wide, but that arm activates two experts and
is not active-compute matched. It is therefore evidence that extra active routing may ease early
optimization, not evidence for the proposed capacity-controlled robustness effect.

The diagnostics sharpen two substrate hypotheses. Frozen linear remains near floor (3.36% OOD,
6.26% ID), so out-of-the-box Cell-DINO features are not sufficient for this exact perturbation
classification protocol. Last-four adaptation gives the strongest ID score (34.51%) but only
12.78% OOD, whereas full adaptation of the original reaches 33.54% ID and 13.77% OOD. This ordering
provisionally favors full representation adaptation for transfer and warns that better ID fitting
does not automatically close the experiment gap. Environment-balanced classification and explicit
output invariance are not helping by epoch 10.

These are trajectories under a 90-epoch schedule, not reproductions of the earlier completed
10-epoch recipe: the horizon changes the learning-rate trajectory. Continued undertraining remains
a live explanation, especially because train accuracy is only 46--53% for most adapted arms and
worst-experiment accuracy is still about 1--2%. The interpretation would be falsified if the
epoch-30/60/90 ordering changes and a fair top-1 MoE opens a substantial OOD advantage while
preserving ID. Until then, the original anchor is the strongest fair instrument and the epoch-90
+5-point gate is unchanged.

Operationally, all ten workers remain healthy at epochs 14--19. A sixth authorized container was
created but is pending because SciServer cannot currently place another two-H100 pod. Its capacity
will shorten the critical path only if it becomes runnable and a predeclared independent action is
licensed; otherwise it must not create an unplanned sweep. The largest current trust limitation is
that the W&B API requested a fresh login, although exact run identities, persistent logs, milestone
files, and OOD-test blindness remain locally verifiable.

## Hypothesis90 pre-epoch-30 handoff: nine arms across, one straggler

Fresh signed W&B pages now resolve every campaign run and repair a provenance-only transcription:
the canonical token-top-1 run is `jo0b8ycc`, not `j0b8ycc`. Nine arms have crossed epoch 30 and
remain running without a rendered fatal trace; the within-experiment-balanced token arm is healthy
but slower at epoch 27. This is operational evidence only. The paired epoch-30 interpretation stays
closed until all ten persistent milestone rows can be parsed and checked for exact identity,
finiteness, `selection_split=ood_val`, and `test_evaluated=false`.

The delayed within-experiment-balanced arm is not a scientific exclusion. Its slower wall time is
compatible with the additional per-environment routing bookkeeping and would be falsified as a
benign straggler if its log stops advancing or a fatal trace appears. The largest immediate trust
threat is temporary loss of a fresh signed SciServer portal session for direct JSON inspection;
W&B confirms live progress but cannot substitute for strict persistent-file validation. OOD test
remains untouched.

## Epoch-30 operational snapshot: possible moderate sparse signal, not yet valid evidence

Every arm has now emitted an epoch-30 W&B milestone. On the rounded log values, canonical token
top-1 is 18.07% OOD validation versus 15.29% for matched dense-wide (`+2.78` points) while also
raising ID from 40.48% to 45.75%. Image top-1 is 18.35%, within-experiment-balanced token is 17.70%,
top-2 is 16.51%, and original is 17.52%. This provisionally weakens the simplest hard-routing
starvation account because top-1 now exceeds top-2, and it keeps routing granularity and
experiment-aware balancing alive. It does not pass the predeclared `+5`-point replication gate.

This is not yet scientific evidence. The values are rounded W&B renderings and the SciServer
session requires reauthentication before the persistent epoch-30 JSON rows can be parsed and
checked at exact precision. The interpretation would be falsified by strict validation failure or
by the epoch-60/90 trajectories returning canonical top-1 to dense-wide. Full train accuracy is
already about 99% for the adapted arms, so remaining changes primarily probe ID generalization and
batch transfer rather than basic task fit. OOD test remains sealed under the last strictly checked
configuration, but its null fields must be reverified in the milestone files before consumption.

## Epoch-60 operational snapshot: persistent moderate average gain, no tail-robustness lift

All ten arms have now emitted epoch-60 W&B milestones. Canonical token top-1 is 19.68% OOD
validation against 17.13% for exact-total-parameter-matched dense-wide (`+2.55` points) and has
higher ID accuracy (47.16% versus 43.61%). This is smaller than the epoch-30 rounded contrast and
below the frozen `+5`-point replication trigger. More importantly, worst-experiment accuracy is
essentially tied (1.54% versus 1.50%), so the current average gain is not evidence of improved
tail robustness.

Routing is not yet the unique explanation. Within-experiment-balanced token routing reaches
19.43% OOD validation, but last-four-block adaptation reaches 19.26% and the original anchor
18.51%; the latter also has the strongest worst-experiment score among these arms at 1.99%.
Top-2 trails top-1, which weakens hard-routing starvation as the dominant failure mode, while the
strong partial-adaptation arm keeps representation depth/forgetting as a competing explanation.
The interpretation would be falsified by strict-file mismatch or a materially different epoch-90
ordering.

This remains operational, not decision-grade. W&B values are rounded and SciServer
reauthentication is required before checking exact persistent JSONs, parameter fairness fields,
`selection_split=ood_val`, `test_evaluated=false`, null OOD-test fields, and the anchor checkpoint.
All ten runs continue healthy; no replication or new sweep is licensed before the epoch-90 gate.

## Final handoff in progress: eight finished, two healthy stragglers

Eight W&B runs have transitioned to `Finished` and expose epoch-90 milestone lines. Matched
dense-wide is still running at epoch 87 and within-experiment-balanced routing at epoch 73, both
with fresh fatal-free logs. The scientific matrix is therefore incomplete and no final contrast
is consumed, even for the canonical token-top-1/dense-wide pair.

This completion frees most of the allocated worker GPUs, but no independent launch is licensed
before the full epoch-90 gate. SciServer reauthentication is also required to validate the eight
result JSONs and checkpoint files before Hugging Face publication. The next meaningful state is
either a failure requiring narrow repair or `10/10` completed arms followed by strict exact-file
validation; OOD test remains sealed.

## Primary epoch-90 pair operationally complete: signal contracts below trigger

Matched dense-wide is now finished, so the primary efficacy pair is available in rounded W&B
form: canonical token top-1 is 20.22% OOD validation versus 18.50% for dense-wide (`+1.72`
points). This is below the predeclared `+5` trigger. Token top-1 preserves ID rather than trading it
away (51.45% versus 48.23%), but its worst-experiment score is slightly lower (1.62% versus 1.70%).

The pattern is not uniquely sparse. Original reaches 20.09% OOD validation, image top-1 20.13%,
environment balancing 19.99%, and last-four adaptation 19.82%. On these rounded values, canonical
token routing is only 0.13 points above original and 0.09 above image routing, while several
noncanonical arms have better worst-experiment scores. This provisionally favors a general
longer-training/adaptation explanation over a large reusable-routing robustness effect.

The formal conclusion remains closed: within-experiment-balanced routing is still running at
epoch 79, and SciServer reauthentication is required to validate exact result JSONs, parameter
counts, split/test-blindness fields, checkpoints, and exclusions. The result would be falsified by
exact-file mismatch or a materially different validated ordering. No replication or mechanism
campaign is licensed from this operational snapshot; OOD test remains sealed.

## Complete epoch-90 operational matrix: substrate rescued, sparse gate likely negative

All ten W&B runs are finished. Canonical token top-1 remains only `+1.72` OOD-validation points
above exact-total-parameter-matched dense-wide, below the `+5` replication trigger. It preserves
ID but is `0.08` points worse on worst-experiment accuracy. Thus the core performance pattern is a
small average gain without tail-robustness evidence.

Within-experiment routing finishes highest on average OOD validation at 20.46%, only 0.24 points
above canonical token routing and 1.96 above dense-wide. Its worst-experiment accuracy falls to
1.38%, 0.32 points below dense-wide. Original and image routing are within 0.13 and 0.09 points of
canonical token routing, respectively, while last-four adaptation and environment balancing are
also close. The complete rounded matrix therefore supports a stronger substrate after 90 epochs,
but not a large, uniquely sparse, or worst-experiment-robust effect.

This is a provisional operational interpretation, not decision-grade evidence. Exact SciServer
files remain inaccessible until the user reauthenticates; the formal negative gate, zero-exclusion
ledger, SHA-256 manifests, and Hugging Face publication must wait for strict validation. No seed
replication or mechanistic campaign is licensed unless validated exact values unexpectedly cross
the predeclared threshold. OOD test remains sealed.

## Validated hypothesis90 decision: competent substrate, no large sparse advantage

SciServer reauthentication resolves the former evidence blocker. Ten of ten exact result JSONs,
40/40 milestone rows, four original-anchor checkpoints, ten manifests, and all campaign logs pass
strict identity, finiteness, coverage, digest, and fatal-error checks. Results share execution
commit/tree `cd783399ab1d4cee2666f1af8dfe3bfd9fc29280` /
`ec05b37b9f2ac593e047243c414b114c3a1fb52c`, seed 0, native CP5 preprocessing, identical data
order, and OOD-validation selection. OOD test is untouched: `test_evaluated=false` and every
withheld-test field is null.

The canonical top-1 token MoE reaches 20.215% OOD validation versus 18.500% for the exact-total-
parameter-matched dense-wide comparator, a 1.715-point gain. It improves ID by 3.213 points but
reduces worst-experiment accuracy by 0.081 points. The 378-parameter difference is 0.001232%, well
inside the frozen 0.1% tolerance. This is below the predeclared 5-point replication threshold, so
the formal gate is negative. The smaller original model reaches 20.093% OOD validation and a
better worst-experiment score; canonical MoE is only 0.122 points higher on average OOD.

The matrix resolves the main competing explanations. The former 10-epoch recipe did understate
substrate strength: all adapted 90-epoch arms fit training perfectly and reach 48--53% ID, while
OOD validation remains 18.5--20.5%. The failure regime is therefore genuine experiment transfer,
not basic representation or optimization failure. Hard top-1 starvation is not supported because
top-2 is worse than top-1. Token granularity is not decisive because image routing is effectively
tied with canonical routing. Global load balancing may suppress specialization on average—the
within-experiment-balanced MoE is highest at 20.459%—but its worst-experiment score falls to
1.380%, below dense-wide's 1.705%, so it does not support tail robustness. Partial adaptation and
environment balancing are close enough to keep general regularization/representation depth as
credible alternatives to sparse conditionality.

The exact checkpoint trajectory also resolves the duration question. Canonical MoE minus
dense-wide OOD validation is -0.416 points at epoch 10, +2.781 at 30, +2.547 at 60, and +1.715 at
90. The qualitative gate is already clear by epoch 60, while the later 30 epochs add 1.58 OOD
points to original, 1.37 to dense-wide, and only 0.54 to canonical MoE. Ninety epochs are therefore
useful to establish the strongest absolute substrate and eliminate undertraining as a critique,
not because the sparse effect grows. The campaign saved model checkpoints only for the original
anchor at epochs 10/30/60/90; exact milestone metrics exist for every arm. A future exploratory
screen can use 60 epochs, with 90 reserved for final confirmation.

This interpretation is decision-grade for the bounded seed-0 target, not proof of zero MoE effect.
It would be overturned by an independently replicated fair contrast exceeding 5 OOD points while
preserving ID and worst-experiment performance. Such replication is not licensed by the frozen
gate. The largest residual uncertainty is single-seed variation; the largest threat to the broad
mechanistic claim is that no valid predictive effect exists to justify gradient-conflict or route-
reuse analyses. The consuming artifact is `analysis/hypothesis90_final_validation.json`; all ten
validated folders are published under the declared Hugging Face prefix and tracked in the fresh
W&B group.

## New exploratory question: factorial60

The user has authorized a new bounded sweep rather than reuse of the failed canonical gate. The
scientific question is now whether the previous canonical choice hid a stronger conditional effect
at a different transformer depth, routing granularity, router geometry, or environment pressure.
The full 3x2x2x3 MoE factorial is paired with six placement/pressure-matched dense-wide controls
and a new original reference on one common native-CP5 seed-0 schedule.

This is an appropriate use of the expanded compute because each arm changes a declared mechanism
and every sparse cell has an explicit fair comparator; it is not evidence that one of 36 searched
cells will generalize. Multiple-search optimism is the largest validity threat. The falsifier is
that no cell reaches the locked +5 OOD-validation-point gain with at most two ID points lost (or a
predeclared consistent worst-experiment improvement). A passing cell is only provisional and must
be rerun unchanged at seeds 1 and 2. OOD test remains unavailable to the sweep.

The immediate implementation is complete but not yet remotely tested or launched. Five running
2-H100 containers are available in the portal; the sixth requested container is pending for lack
of an eligible H100 pair. Job-level GPU freedom, focused/full test success, clean execution commit,
and dry-run identity are the remaining launch gates.

## 2026-08-01 21:19 ET — factorial60 execution handoff

The launch gate passed. Source commit `bd213dbd7758f456eb822379707627b1998847ff` and isolated
SciServer execution commit `b8ece25e05dc675bd6a61e0728879e53130e453e` share exact tree
`7320d64944c34c9ee832924ee06429491739354f`; the remote pinned environment passes 30 focused and
106 full tests. Dry runs prove five nonoverlapping shards with 43 total cells, and prelaunch audits
prove no duplicate process/result and ten free owned GPUs.

All five launchers are active with two distinct workers per container. Ten fresh epoch-0 JSONL
records and per-GPU model allocations verify training rather than mere launcher/PID presence; the
fatal scan is empty. Current coverage is 0/43 strictly valid, 10 active, 33 queued, and 0 available
H100s idle. Ten tracking directories were created with the declared W&B group
`rxrx1-cell-dino-factorial60-20260801`, job type `rxrx1_factorial60`, and OOD-test-blind tags. HF
upload is pending strict completion validation by design.

This is operational progress only. Training loss and epoch-0 activity carry no scientific claim.
The largest trust threat is search multiplicity across 36 MoE cells; the antidote remains locked
matched controls plus unchanged paired-seed replication for any promoted cell. The next licensed
action is to validate each 10-epoch milestone, continue/prune under the declared rule, and refill
each released GPU immediately from the remaining queue. `tester6` remains pending for the recorded
scheduler capacity reason, so no sixth-container work is yet possible.

## 2026-08-01 22:02 ET — first factorial60 successive-halving handoff

The first ten searched MoE cells now have complete, strictly validated epoch-10 milestones. Exact
rows pass identity, finiteness, environment-coverage, split, test-blindness, ERM-objective,
provenance, tracking, and fatal-log checks. Early token-cosine canonical is highest on mean OOD
validation (`0.134869`), followed by early image-linear route (`0.133245`); the latter has the
stronger worst-experiment score (`0.018263`). The smallest OOD spread among these searched cells is
not an efficacy contrast, because no placement/pressure-matched dense control has yet reached
epoch 10.

Applying the predeclared strict Pareto rule across OOD validation, ID retention, and worst-
experiment accuracy removes four dominated cells and preserves six nondominated cells. The four
released H100 leases immediately started four new, nonduplicate cells, restoring 10/10 active GPU
workers with 29 queued. Fresh tracking directories, exact group/job/tags, roughly 8 GiB model
allocations, and empty fatal scans verify the refill. The normalized evidence and exact triage list
are in `analysis/factorial60_epoch10_validation.json`.

The current evidence provisionally favors cosine token routing for mean OOD and image-linear route
pressure for the mean/tail tradeoff, but the sharp falsifier is their exact matched dense controls:
if those controls close the gap, the apparent advantage is placement/pressure or ordinary capacity,
not conditional computation. Multiplicity remains the largest threat, and any eventual winner is
exploratory until locked seeds 1 and 2 replicate it without OOD-test access.

## 2026-08-01 23:08 ET — factorial60 wave-two successive-halving handoff

Four new epoch-10 milestones and six epoch-30 milestones are strictly valid. Cumulative coverage is
14 valid epoch-10 rows and six valid epoch-30 rows, with no completed result JSON yet. Early
image-linear route is the sole epoch-30 frontier survivor: OOD validation `0.200223`, ID `0.475672`,
and worst experiment `0.019075`. It strictly dominates the five other first-wave survivors on all
three predeclared triage metrics. Middle image-linear output is the sole new epoch-10 survivor at
OOD `0.129186`, ID `0.308923`, and worst experiment `0.018263`.

Three new epoch-10 cells and five epoch-30 cells were pruned only after complete validated
milestones. Eight nonduplicate cells immediately refilled the released leases. Ten direct workers
remain active across five 2-H100 containers, 21 cells remain queued, 12 are cumulatively pruned,
and zero of the ten available allocations are idle. All live workers use the declared W&B
group/job/tags; five fatal scans are empty; the isolated execution checkout is clean at
`b8ece25e05dc675bd6a61e0728879e53130e453e` / tree
`7320d64944c34c9ee832924ee06429491739354f`. `tester6` remains Pending for the exact recorded
scheduler-capacity reason after one start request.

The exact dense baselines and their queue locations are known. The present leader pairs with
`rxrx1_dense_wide_early_canonical_E8_ep60_s0_factorial60_20260801`; its matched 60-epoch-schedule
milestone is pending, rather than unknown. The older middle-canonical dense trajectory is valid
context but not an exact replacement because placement and the 60-versus-90-epoch learning-rate
schedule differ. Therefore the current result is diagnostic/provisional only. Its sharp falsifier
is the locked early canonical dense contrast: less than +5 absolute OOD points without a consistent
tail improvement rejects a material sparse effect. Multiplicity remains the largest threat. OOD
test is sealed.

## 2026-08-01 23:26 ET — factorial60 wave-three successive-halving handoff

Eight more epoch-10 milestones pass exact identity, finite-metric, four-environment coverage/count,
ERM, provenance, tracking, fatal-scan, `selection_split=ood_val`, and `test_evaluated=false` checks.
Cumulative coverage is 22 valid epoch-10 and six epoch-30 rows. Early token-cosine route is the only
new frontier cell: OOD validation `0.136391`, ID `0.317615`, and worst experiment `0.013393`. Seven
other cells are strictly dominated on all three frozen triage axes and were pruned after validation.

Seven distinct cells immediately refilled the leases, leaving 10 active, 14 queued, 19 cumulatively
pruned, 29 W&B runs launched, and zero idle among the ten available H100s. The refill includes the
early output-pressure dense-wide arm; the separate early canonical dense comparator required for the
current route-pressure leaders is known and remains queued. All new direct workers are owned, two
model allocations are present per container, and their environments carry the declared W&B
group/job/tags. `tester6` remains Pending for the recorded scheduler-capacity reason after one start
request.

The result is a multiplicity-exposed seed-0 screening signal. A plausible interaction between early
placement, token routing, cosine geometry, and within-experiment route pressure is now worth carrying
to epoch 30, but ordinary capacity or early optimization can still explain it. Its falsifier is the
locked matched-dense contrast and tail behavior at later milestones. OOD test is sealed.

## 2026-08-01 23:40 ET — next-family architectural hypotheses registered

The factorial60 evidence set did not change: 22 epoch-10 and six epoch-30 rows remain strictly
valid, with zero completed result JSONs. Five independent container audits still show ten direct
owned workers, two model allocations per container, clean execution commit
`b8ece25e05dc675bd6a61e0728879e53130e453e`, and no fatal file. `tester6` remains Pending for the
same exact scheduler-capacity reason; no duplicate request was made.

The new `analysis/architectural_hypothesis_backlog.md` turns a primary-literature synthesis into 24
predeclared causal tests. The ranking favors function-preserving sparse upcycling, router stability
and pressure, expert-choice/soft/shared-expert routing, gradient-conflict-localized placement, and
tail-safe robust objectives because those directly target the current early mean-OOD signal, weak
tail performance, and the prior epoch-30/60 gain followed by epoch-90 shrinkage. Each design has a
matched dense or router control and a stated falsifier; active-compute and total-parameter
estimands remain separate.

The registry is planning evidence, not a run result. The 14-cell factorial60 queue remains the only
currently tested runnable refill set. The highest-priority scientific gate is still the matched
early-canonical dense milestone. Next-family arms become runnable only after isolated
implementation, parameter/FLOP accounting, focused/full tests, dry-run identity, no-duplicate
checks, persistent destinations, and OOD-test-blind validation pass.

## 2026-08-02 00:11 ET — factorial60 wave four and first validated final result

Seven new epoch-10 rows, middle image-linear output at epoch 30, and early image-linear route at
epoch 60 pass strict JSON, identity, finite-metric, four-environment/count, ERM, provenance,
tracking, fatal-scan, `selection_split=ood_val`, and `test_evaluated=false` checks. The completed
route arm additionally has all held-out-test metrics null. Cumulative validated coverage is now 29
epoch-10, seven epoch-30, one epoch-60, and one final result.

Early image-linear route finishes at train/ID/OOD-validation/worst-experiment
`1.000000/0.527529/0.211082/0.024756`; all three held-out metrics improve from epoch 30. The first
new dense control, early output pressure, leads the entire epoch-10 screen on mean OOD validation
at `0.144510`. That is diagnostic evidence that ordinary dense capacity and training pressure can
explain early apparent sparse advantages. It is not the leader's exact comparator: route-pressure
MoE still pairs with the queued early-canonical dense arm.

Five new epoch-10 MoE cells are strictly dominated by the existing early image-linear route
frontier. Middle image-linear output at epoch 30 is strictly dominated by the previously validated
early token-linear output epoch-30 row. Those six live workers were stopped after preserving their
rows. The completed worker exited cleanly, and all seven leases immediately started distinct next
cells. The active set is ten, the tested queue is seven, cumulative pruning is 25, and there are no
unassigned available H100s. The first completed folder has a four-file SHA-256 manifest plus the
manifest itself; the five-file HF prefix was uploaded and remotely re-listed.

This remains a multiplicity-exposed seed-0 architecture screen. A sparse efficacy claim is
falsified unless the early image-linear route arm beats its exact early-canonical dense control on
the locked 60-epoch schedule by at least five absolute OOD-validation points without losing more
than two ID points, or produces the separately predeclared consistent worst-experiment gain.
`tester6` remains Pending for the unchanged 11-affinity, 2-unschedulable, 3-insufficient-GPU
scheduler reason; no duplicate start was issued. OOD test is sealed.

## 2026-08-02 00:29 ET — route-pressure epoch-30 mechanism signal

The protected early token-cosine route-pressure arm now has a strictly valid epoch-30 milestone:
train/ID/OOD-validation/worst `0.999906/0.467596/0.194134/0.021510`. Cumulative factorial60
coverage is 29 epoch-10, eight epoch-30, one epoch-60, and one completed result. The arm is second
on epoch-30 mean OOD and first on epoch-30 worst-experiment accuracy, so it remains on the frozen
Pareto frontier and continues to epoch 60.

The matched routing-pressure contrast is directional: versus early token-cosine canonical MoE,
route pressure is `+0.004364` OOD, `-0.006993` ID, and `+0.006088` worst experiment. This isolates a
possible pressure/geometry interaction but does not show that MoE beats dense. The exact
early-canonical dense comparator is known and queued; ordinary early-placement capacity and
optimization remain viable alternatives. The sharp falsifier is loss of the mean/tail signal at
epoch 60 or its disappearance against that locked dense comparator.

Strict JSON, run/seed/epoch/config, finite metric, four-environment, ERM, tracking, fatal-scan,
`selection_split=ood_val`, and `test_evaluated=false` checks pass. No OOD-test metric exists. The
worker was healthy beyond epoch 37, all ten available H100s were assigned, seven tested cells were
queued, and tester6 remained Pending for the recorded capacity reason. Search multiplicity and
seed-0 selection remain the largest threats.

## 2026-08-02 00:38 ET — dominated late output cell released

The late image-linear output arm passes strict epoch-10 validation at OOD/ID/worst
`0.119444/0.293386/0.015422` and is strictly dominated by the validated middle token-cosine
canonical cell on all frozen triage axes. It was pruned after preservation. The shard scheduler
refilled GPU 0 on container 2875 with late token-linear route (PID 73667), which acquired 7,997 MiB
without a duplicate output or process. Coverage is 30/8/1, with 26 pruned, 10 active, six queued,
37 W&B launches, and no unassigned available H100. This is allocation evidence only; OOD test is
sealed and exact dense comparisons remain the efficacy gate.

## 2026-08-02 00:59 ET — wave seven completes the predeclared launch matrix

Eight milestones are strictly valid: six epoch-10 and two epoch-30 rows, taking cumulative coverage
to 36/10/1 with one final result. Original Cell-DINO and late image-linear route remain
Pareto-relevant at epoch 10. Six dominated rows were pruned after preservation. Most importantly,
middle token-cosine canonical's early tail signal disappears by epoch 30: early token-cosine route
is better by `+0.009742` OOD, `+0.014602` ID, and `+0.002841` worst experiment. Early token-cosine
route also dominates early output dense at epoch 30. The provisional mechanism ranking therefore
favors early routing pressure; ordinary capacity and optimization remain live alternatives.

All 43 planned arms have now launched. One manual priority refill raced a precomputed shard queue,
creating a second physical early-canonical dense process. The duplicate was found after W&B startup
but before epoch 0, stopped, and excluded from every scientific count; the auto-launched comparator
was retained. Queue controllers for shards 0 and 1 were stopped while their child workers remained
healthy, preventing later duplicates. The remaining late token-linear canonical and late
token-cosine output arms then launched from the frozen cell registry. The scientific denominator is
43 unique arms; physical W&B starts are 44 with one explicit zero-milestone exclusion.

All ten available H100s hold distinct active arms, the runnable factorial queue is exhausted because
launch coverage is complete, and tester6 remains Pending for the recorded scheduler-capacity reason.
OOD test is sealed. The sharp falsifier is the exact early-canonical dense trajectory matching or
exceeding the early routed arms, or loss of the routed mean/tail signal at epoch 60. Multiplicity and
seed-0 selection are the largest current threats.

## 2026-08-02 05:16 ET — factorial60 final decision and router-aux continuation

All nine factorial60 epoch-60 results pass strict result, milestone, log, checkpoint, parameter,
provenance, four-environment, split-blindness, and finite-metric validation. All nine have manifests
and verified HF folders. The complete seed-0 screen contains 43 unique scientific arms, 34
predeclared prunes, nine finals, and one separately excluded pre-epoch-0 duplicate physical start.
No OOD-test field was populated or consumed.

Late canonical dense-wide leads mean OOD validation (`0.214735`). Early image-linear route is the
best sparse finalist (`0.211082`) and trails it by `0.365` OOD and `2.071` ID points while improving
worst-experiment accuracy by `0.406` points. Against the same-placement early dense control, it
improves OOD/ID/worst by `1.847/3.218/0.933` points. This is a real directional placement-matched
signal, but not the `+5` mean-OOD effect and not a fresh-seed replication. The leading explanation
is a depth/placement effect with a possible sparse tail tradeoff; the falsifier is consistency under
locked fresh seeds or a router setting that beats the best dense arm without sacrificing ID.

Three router-aux finals pass the same split and artifact checks and are published. Zero balance plus
z-loss `1e-3` leads at OOD `0.212401`; it improves the factorial token-route reference by only
`0.436` points, remains `0.233` below the best dense arm, and worsens the tail. Ten nonduplicate
settings are active and three remain queued across the five running 2-H100 containers. The active
remote execution commit `7dcb42e` contains the physical-occupancy refill guard and is clean. The
local source repair additionally adds restart markers. Four earlier oversubscription attempts are
operationally excluded before scientific milestones.

The sixth container was stopped rather than pending at inspection. One start request was issued as
authorized and the immediate portal recheck still showed stopped; no duplicate request was made.
Classification is `RUNNING_HEALTHY`: all running H100s have decision-relevant nonduplicate work,
completed artifacts are persisted, and the next useful action depends on new router milestones.

## 2026-08-02 05:45 ET — router auxiliary epoch-10 validation and refill

Thirteen router-auxiliary epoch-10 rows are strictly valid. Zero load balance under route pressure
leads mean OOD validation (`0.138827` with z-loss `0.001`; `0.138522` without z-loss), but the two
rows differ by only `0.000304`, so z-loss is not the isolated cause. The best tail is instead the
canonical `balance=0.01,zloss=0` row (`0.017045`) at lower mean OOD (`0.130505`). The evidence
therefore favors a mean-versus-tail load-balance tradeoff at epoch 10, not a general router-auxiliary
win. Epoch-30 persistence without ID or worst-experiment loss is the falsifier.

Three strict Pareto prunes released exactly the three shard-local tested refills, which launched
without a duplicate: canonical `balance=0.001,zloss=0`, `balance=0.01,zloss=0.01`, and
`balance=0,zloss=0`. Ten distinct workers again occupy the ten H100s in the five running containers;
all ten logs initialized W&B and fatal scans are clear. The full screen now accounts for three
validated finals, three pruned rows, and ten active rows, with no queued router cells. Two dominated
but pressure-paired `balance=0.001,zloss=0.001` rows remain licensed to epoch 30 for the declared
route-pressure interaction trajectory; they do not count as frontier winners.

The exact execution remains clean `7dcb42e` / tree `1755b2f`; every validated row uses OOD
validation only and leaves OOD test sealed. `tester6` remained stopped after the single authorized
start request in this invocation. Multiplicity and seed-0 selection are the largest scientific
threats.

## 2026-08-02 06:18 ET — router balance signal changes by epoch 30

The strict router-auxiliary ledger now contains 25 valid milestone rows across all 16 registered
streams: 13 epoch-10, nine epoch-30, and the three previously validated epoch-60 finals. Six
epoch-30 rows are new. Every consumed row has exact run/config/seed/epoch identity, finite metrics,
four validation environments totaling 9,854 samples, ERM, a declared checkpoint, clean execution
provenance, `selection_split=ood_val`, `test_evaluated=false`, and no OOD-test value. The normalized
record is `analysis/router_aux60_epoch30_validation.json`.

The epoch-10 low-balance lead is an early transient rather than a stable ordering. At epoch 30,
route `balance=0.01,zloss=0` leads mean OOD (`0.198295`). Its exact canonical-pressure pair reaches
`0.186016`, so the route-pressure delta is `+1.228` OOD points, `-0.207` ID points, and `+0.081`
worst-experiment points. A second exact pair at `balance=0.0001,zloss=0.001` favors route pressure
by `+0.548/+0.867/+0.325` points. Conversely, zero balance without z-loss leads the tail
(`0.019075`) but is `0.609` mean-OOD points below the overall epoch-30 leader. Moderate balance and
route pressure are now the most plausible interaction; neither zero balance nor z-loss alone is
supported.

Three canonical rows that are globally dominated remain licensed because each is the exact
pressure control for a routed setting and therefore changes the causal interpretation. The other
live rows are Pareto-relevant or complete the remaining matched pair; pruning them would release
compute before a tested next-family refill exists without answering a sharper question. Ten
workers remain active and zero H100s in the running five-container pool are unassigned. The three
latest refills still await epoch 10. The falsifier is loss of the matched route-pressure advantage
or an ID/tail penalty at epoch 60. Seed-0 multiplicity remains the largest threat and OOD test is
sealed.

## 2026-08-02 06:26 ET — complete epoch-10 coverage sharpens the interaction

All 16 router-aux settings now have a strict epoch-10 row, ten have strict epoch-30 rows, and the
three prior finalists retain strict epoch-60 rows: 29 validated milestones total. The last
epoch-30 pressure pair, `balance=0.001,zloss=0.001`, changes OOD/ID/worst by
`-0.193/+0.153/-0.406` percentage points for route versus canonical pressure. Route pressure is
therefore not beneficial in general; its positive signal is conditional on auxiliary weights.

The three new canonical epoch-10 OOD/ID/worst rows are: zero balance and zero z-loss
`0.135174/0.310598/0.013393`; balance `0.01` and z-loss `0.01`
`0.125127/0.297572/0.011769`; and balance `0.001` with zero z-loss
`0.131825/0.308308/0.015828`. The high-z-loss row is strictly dominated, but it remains licensed
only to test the predeclared delayed router-stabilization alternative at epoch 30; the other two
are exact matched controls. The scientific claim stays exploratory and multiplicity exposed.

## 2026-08-02 07:52 ET — route-pressure persistence fails at epoch 60

Nine router-aux cells now have strict epoch-60 results, bringing campaign coverage to
16/13/9 at epochs 10/30/60 and 38 validated milestone rows. Every consumed result has exact
registry/config/seed identity, finite train/ID/OOD-validation/worst metrics, four environments and
9,854 samples, ERM, the declared 30,676,212 parameters, a valid epoch-60 checkpoint, clean
`7dcb42e` provenance, `selection_split=ood_val`, `test_evaluated=false`, and no OOD-test value. Nine
manifests are present and all nine completed HF folders were remotely re-listed. The normalized
record is `analysis/router_aux60_epoch60_validation.json`.

The moderate-balance pressure effect is transient: route minus canonical at
`balance=0.01,zloss=0` falls from `+1.228` OOD points at epoch 30 to `+0.142` at epoch 60, with
`+0.401` ID and `+0.365` worst-experiment points. A second completed pair instead trades
`-0.335` mean OOD for `+0.446` worst-experiment points. Router pressure and auxiliary losses can
move the mean/tail trajectory, but no completed pair supports the sought large stable mean-OOD
gain. This is exploratory seed-0 mechanism evidence, not confirmation.

The next bounded question tests whether the learnable cosine router's initial temperature causes
that transient. Twelve cells are registered in `scripts/sweep_rxrx1_router_temperature.py`; six
pressure-paired cells are running and six remain queued. They change no parameter count or active
compute. Four remaining router-aux workers plus the six new cells occupy all ten running H100s.
`tester6` received one start request while stopped and was not duplicated. The sharp falsifier is
absence of a consistent temperature-by-pressure gain at epochs 30/60; any apparent winner remains
multiplicity exposed until locked fresh-seed confirmation. OOD test is sealed.

## 2026-08-02 09:12 ET — tail-safe sparse survivor licenses locked fresh-seed pairs

The router-auxiliary campaign is closed at 13 valid finals, three preserved prunes, and complete
16/13/13 epoch-10/30/60 coverage. The final normalized record is
`analysis/router_aux60_final_validation.json`. All consumed rows pass exact identity, finite metric,
four-environment and 9,854-count, ERM, parameter, checkpoint, clean `7dcb42e` provenance,
fatal-scan, OOD-validation selection, and OOD-test-blind checks. Thirteen manifests exist and the
strict batch publisher reports success for the four newly completed folders.

Canonical pressure at `balance=0.01,zloss=0.01` is the locked seed-0 survivor. Against the exact
same-placement early dense comparator its OOD/ID/worst deltas are `+1.979/+3.130/+0.649` points;
against the best late dense they are `-0.233/-2.159/+0.122`. This rules out a large `+5` mean-OOD
effect for this recipe, but licenses the predeclared smaller-mean plus consistent-tail pathway.
The sharp falsifier is failure of the locked sparse-minus-exact-dense tail direction at seeds 1
and 2, or a pooled mean interval compatible with no useful OOD benefit. No tuning is allowed inside
that confirmation.

The initial temperature handoff contains six strict epoch-10 rows. Paired route-minus-canonical
OOD deltas are `0.000`, `-0.477`, and `+0.041` points, so initialization temperature has not created
a material early ordering. Four distinct temperature refills launched into the four released GPU
slots; ten temperature workers now occupy all ten running H100s, with two cells queued. The tested
ready queue contains 14 arms: four locked seed-1/2 confirmation cells, eight 4-versus-16 expert
architecture cells, and the two remaining temperature cells. All passed 18 focused tests and dry
runs in isolated clean checkout `4893c964` / tree `74e62d3b`; source commit is `d592a89`.

The largest scientific threat is winner selection across the prior 43-arm factorial and 16-cell
router-aux screen. The largest operational tracking uncertainty is that declared W&B groups were
set in launch commands but no local console marker or live environment key was recoverable; remote
W&B state is not claimed. Persistent logs, milestones, results, checkpoints, and HF manifests are
intact. OOD test remains sealed.

## 2026-08-02 10:14 ET — zero-auxiliary temperature evidence remains directional

Strict router-temperature coverage is now 11 epoch-10, ten epoch-30, and one epoch-60 row. The
new high-temperature zero-auxiliary canonical row reaches epoch 10 at OOD/ID/worst
`0.133956/0.307963/0.014610`, while the low-temperature canonical row reaches epoch 30 at
`0.189263/0.466931/0.017045`. Its exact route-pressure pair reaches epoch 30 at
`0.189872/0.460677/0.017045`, while high-temperature canonical plus z-loss reaches epoch 60 at
`0.211792/0.530681/0.019886`. All 22 cumulative rows pass exact identity, finite-metric,
four-environment/9,854-sample, ERM, checkpoint, clean `7dcb42e` provenance, fatal-scan,
OOD-validation-selection, and OOD-test-blind checks.

Within the zero-auxiliary canonical branch at epoch 10, temperature `0.2` versus `0.03` changes
OOD/ID/worst by `-0.284/-0.485/+0.284` points. At temperature `0.03`, removing the small z-loss at
epoch 30 changes the same metrics by `-0.548/-0.790/+0.041` points. The latest evidence therefore
keeps a low-temperature mean/ID optimization direction alive, but with an opposing tail movement.
The exact zero-auxiliary low-temperature route-minus-canonical contrast is
`+0.061/-0.625/+0.000` OOD/ID/worst points, which rules out a material sparse effect for that cell.
The first epoch-60 row is `+1.918/+3.533/+0.446` points over exact early dense, but its mean and tail
are below the already locked router-aux survivor, so it does not open another fresh-seed branch.
The epoch-30 zero-auxiliary temperature comparison and complete epoch-60 pairs remain the sharp
falsifiers.

Ten distinct workers occupy all ten running H100s across containers 2887, 2875, 2874, 2862, and
2859; all resolve to clean execution commit/tree `7dcb42e` / `1755b2f` and have fresh fatal-free
logs. One temperature cell remains queued, followed by four locked fresh-seed confirmation arms
and eight expert-count arms; ready/backlog counts remain 13/24. `tester6` received one fresh start
request and remained stopped. No final result, manifest, or HF folder exists. OOD test is sealed.

## 2026-08-02 09:38 ET — lower temperature is directional, not yet a sparse gain

Strict validation now covers ten epoch-10 and six epoch-30 router-temperature rows. Every consumed
row has exact registry/run/seed/milestone identity, finite metrics, ERM, four environments and 9,854
OOD-validation samples, exact parameter accounting, a saved checkpoint, clean `7dcb42e` provenance,
no fatal log, `selection_split=ood_val`, `test_evaluated=false`, and no OOD-test fields.

Under `balance=0,zloss=0.001`, lowering initial router temperature from `0.2` to `0.03` improves
epoch-30 OOD validation for canonical and route pressure. However, route minus canonical is still
`-0.842` points at `0.2` and `-0.629` points at `0.03`. The exact route `0.2` cell is also worse on
ID and worst-experiment accuracy, so it was pruned after preserving its 368,514,056-byte epoch-30
checkpoint. This rules out that specific high-temperature route-pressure recipe; it does not rule
out the lower-temperature optimization direction or interactions with the unfinished zero-auxiliary
pairs. The sharp falsifier is disappearance at epoch 60 or non-reproduction in zero-auxiliary pairs.

Container 2874 GPU0 immediately refilled with canonical `temperature=0.2,balance=0,zloss=0`
(PID 47384); the duplicate/provenance preflight passed and live W&B run `rc731n1b` was verified.
All ten running H100s are assigned across five containers. Temperature accounting is ten active,
one queued, one pruned, zero finals; the ready queue remains 13 with a 24-question backlog.
`tester6` 2893 remains stopped after one start request. This remains exploratory seed 0 with
12-cell multiplicity, and OOD test is sealed.

## 2026-08-02 10:00 ET — temperature ordering is auxiliary-setting dependent

Two new exact `balance=0.01,zloss=0` epoch-30 rows pass the strict registry, identity, finite metric,
ERM, four-environment/9,854-sample, parameter, checkpoint, clean `7dcb42e` provenance, fatal-scan,
OOD-validation-selection, and OOD-test-blind checks. Router-temperature coverage is now 10/8/0 at
epochs 10/30/60; there are no final results or manifests.

Temperature `0.2` now beats `0.03` on mean OOD for both canonical (`+0.335` points) and route
(`+0.467` points) at moderate balance, reversing the zero-balance plus z-loss direction. At
temperature `0.2`, route minus canonical changes OOD/ID/worst by `+0.792/+0.399/-0.244` points.
This is a directional mean/ID sparse effect with a tail tradeoff, not a tail-safe gain. It rules out
a global lower-temperature benefit and makes a temperature-by-auxiliary-pressure interaction most
plausible. Transient optimization noise or a loss-only effect remains viable; persistence at epoch
60 and a consistent zero-auxiliary architecture contrast are the falsifiers. Neither row is pruned.

All five running containers expose two distinct active workers from the clean dedicated
`7dcb42e` / `1755b2f` execution checkout, with zero idle running H100s and no fatal marker. The
shared persistent project checkout is dirty legacy state but is not an execution tree. Tracking
markers are initialized without errors; HF publication remains correctly empty before finals.
`tester6` 2893 stayed stopped after exactly one start request. One temperature cell remains queued,
followed by four locked fresh-seed confirmation arms and eight expert-count cells; ready/backlog
counts remain 13/24. OOD test remains sealed and 12-cell seed-0 multiplicity is explicit.

## 2026-08-02 10:23 ET — low-temperature canonical plus z-loss is the new provisional survivor

The first complete epoch-60 pressure pair raises strict router-temperature coverage to 11/10/3.
At `temperature=0.03,balance=0,zloss=0.001`, canonical OOD/ID/worst are
`0.218693/0.527750/0.022727` and route is `0.204485/0.526962/0.016234`. Route minus canonical is
`-1.421/-0.079/-0.649` points, a clean negative for within-experiment route pressure in this cell.

Against exact early dense, the canonical row is `+2.608/+3.240/+0.731` OOD/ID/worst points and it
strictly exceeds the prior locked router-aux survivor by `+0.629/+0.111/+0.081`. This is a stronger
exploratory seed-0 candidate under the predeclared smaller-mean plus tail pathway, but not a
confirmation. The gain now points toward global auxiliary regularization or temperature-dependent
router geometry rather than route-pressure specialization. Fresh-seed exact-dense comparisons are
the sharp falsifier after the temperature campaign closes; multiplicity remains the largest threat.
All 24 cumulative rows pass strict checks and OOD test remains sealed.

## 2026-08-02 10:50 ET — five strict finals, full refill, and next architecture split

Temperature coverage is now `11/10/5`; five exact result folders pass strict final validation,
have manifests, and are remotely listed with seven HF files each. At
`temperature=0.03,balance=0.01,zloss=0`, route minus canonical is
`+0.162/-0.059/+0.284` OOD/ID/worst points. This is directional but smaller than the provisional
low-temperature canonical plus z-loss leader and is not promoted.

The last temperature arm and locked seed-1/2 sparse/dense pairs now occupy the four GPUs released
by completed temperature runs. Exact assignments are recorded in `.steward/state.json`; ten
workers occupy ten H100s and none are idle. A registry-availability launch failure occurred before
model start and was repaired once with the exact frozen commands from clean execution commit
`7dcb42e7`; no scientific run was excluded.

The ready queue is being replenished with two sparse fresh-seed leader confirmations sharing the
active dense anchors and four active-compute-matched E4/E16 pressure cells at the winning router
temperature/z-loss. The E8 pressure pair is shared, not duplicated. These six arms remain pending
remote tests and dry runs; after those checks they combine with eight tested expert-count arms for
14 runnable candidates. The decisive falsifiers are fresh-seed sparse-minus-dense consistency and
loss of the E8 signature under E4/E16. OOD test remains sealed.

Remote licensing is complete: all five copied source/test files match GitHub `f5ac9f4` by SHA-256,
five focused tests and all 118 tests pass, and dry runs enumerate exactly two locked fresh-seed
sparse arms and four E4/E16 pressure arms. These six join the eight previously tested expert-count
cells, making 14 runnable arms. The test checkout is explicitly code-equivalent on base execution
commit `7dcb42e`; it is not mislabeled as a clean checkout of the unreachable GitHub object.

## 2026-08-02 11:06 ET — high-temperature moderate balance is a mean tie with a tail cost

Router-temperature validation now covers `12/11/7` strict epoch-10/30/60 rows. At
`temperature=0.2,balance=0.01,zloss=0`, canonical OOD/ID/worst are
`0.206515/0.526421/0.020698` and route is `0.206921/0.526864/0.018263`; route minus canonical is
`+0.041/+0.044/-0.244` points. The mean and ID effect has effectively vanished by epoch 60 while
the tail cost persists. This rules out a robust moderate-balance route-pressure gain for the exact
pair and leaves global auxiliary regularization or temperature-dependent router geometry as the
more plausible explanation for the seed-0 leader.

All 30 rows pass exact identity, finite-metric, four-environment/9,854-sample, ERM, checkpoint,
clean-provenance, fatal-scan, OOD-validation-selection, and OOD-test-blind checks. The two workers
remain allocated while final-result packaging completes, so there is no idle GPU to hand off yet.
The next two releases are licensed for locked sparse leader seeds 1 and 2 sharing the already
active exact-dense anchors, followed by E4/E16 pressure arms. Multiplicity remains explicit and
OOD test remains sealed.

## 2026-08-02 11:14 ET — strict publication and locked fresh-seed handoff

The completed `temperature=0.2,balance=0.01,zloss=0` pair now has strict result validation,
manifests, and seven remotely verified HF files per run. This raises temperature finals/manifests
to `7/7` and the remote verified-file count to 49. The scientific interpretation is unchanged:
route pressure is a mean/ID tie with a tail cost at epoch 60, not a promotion.

The two released GPUs were immediately assigned to the predeclared low-temperature canonical
sparse leader at seeds 1 and 2. Container 2874/GPU1 runs worker 49277 (W&B `po7ckidl`) and
container 2862/GPU0 runs worker 48824 (W&B `9sjnn6el`), paired with already-active exact dense
anchors. A post-launch shell-scope error affected only the `.active` marker writes; both detached
workers were healthy, and the markers were narrowly repaired once to their actual GPU PIDs. All
ten running H100s are occupied by distinct arms, 12 tested arms remain ready, and OOD test stays
sealed.
