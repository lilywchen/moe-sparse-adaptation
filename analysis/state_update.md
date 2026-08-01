# Living scientific state

Last verified: 2026-08-01 05:14 EDT

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
mechanism, replication, or OOD-test experiment is licensed.
