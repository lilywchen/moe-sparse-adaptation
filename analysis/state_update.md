# Living scientific state

Last verified: 2026-07-31 23:15 EDT

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

1. Complete and strictly validate the two active Cell-DINO full-fine-tuning diagnostics, then
   determine whether the current failure is representation/optimization, ordinary generalization,
   or batch transfer.
2. If dense competence is established, freeze one recipe and run the seed-0 original versus exact
   total-parameter-matched dense-wide versus canonical-MoE kill contrast.
3. Replicate only if MoE gains at least 5 absolute OOD-validation points while losing no more than
   2 ID points; prioritize the 10--15 point effect target.
