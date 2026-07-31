# Living scientific state

Last verified: 2026-07-31 18:58 EDT

## Scientific question

When a pretrained vision transformer is fully adapted to scientific images collected across
acquisition environments, does sparse conditional capacity improve held-out-environment
generalization over a total-parameter-matched dense model, and which routing choices explain the
gain or failure?

## Where the project is

The study is in Stage 0. The frozen 36-cell factorial has not started. All six RxRx1 seed-0 shared-
HPO candidates have completed and passed strict validation under execution commit `26ad7fa`.
One Camelyon17 candidate is valid, giving `7/12` formal results overall; three Camelyon17 workers
remain active and the launcher retains the remaining queue.

RxRx1 remains behind a substrate sanity gate. The canonical WILDS ResNet-50 ERM reproduction is at
epoch 22, and two bounded DINOv2 rescue diagnostics are now running: removing random-resized crop
and removing layer-wise LR decay, one factor at a time. They use validation/ID evidence only; the
OOD-test subset remains unavailable.

## Decision-grade findings

No MoE-versus-dense performance finding is decision-grade yet. The RxRx1 grid is complete and can
be ranked, but its sanity gate remains open because all six models occupy an inadequate accuracy
regime. There is not yet a valid basis for freezing an RxRx1 recipe, selecting a Camelyon17 recipe,
comparing MoE with dense controls, or making a mechanism claim.

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

The two rescue runs isolate leading explanations from the fixed best DINOv2 anchor `(lr=1e-4,
LLRD=0.85)`: `rxdiag_no_rrc` changes only random-resized cropping, and `rxdiag_uniform_lr` changes
only LLRD to 1.0. Both are diagnostic-only and cannot enter the formal HPO ranking.

## Current scientific interpretation

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

The canonical curve has resolved the main data-pipeline ambiguity: RxRx1 is learnable here. The two
single-factor diagnostics now distinguish whether the failure is primarily caused by destructive
cropping or by suppressing early-layer adaptation. A large rescue from either makes a corrected
DINOv2 protocol plausible; a null result from both makes backbone replacement the principled next
move.

## Implications for experimental design

- Record the formal RxRx1 ranking, but do not replicate or freeze it until the sanity gate resolves.
- Do not change factorial factors, levels, losses, or selection rules in response to this diagnostic.
- Camelyon17 may advance independently when its six valid seed-0 candidates complete; it need not
  wait for the RxRx1 diagnosis.
- Treat routing, decodability, and embedding analyses as mechanism evidence only after the matched
  predictive comparison is valid.

## Next decisions

1. Complete and validate `rxdiag_no_rrc` and `rxdiag_uniform_lr` against the fixed anchor.
2. If one strongly rescues DINOv2, freeze that correction and repeat the shared recipe screen; if
   neither does, make an explicit backbone change before spending the Stage-1 budget.
3. Complete and rank Camelyon17 Phase A independently.
4. Freeze shared recipes and calibration values before launching any Stage-1 factorial cells.
