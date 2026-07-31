# Living scientific state

Last verified: 2026-07-31 18:12 EDT

## Scientific question

When a pretrained vision transformer is fully adapted to scientific images collected across
acquisition environments, does sparse conditional capacity improve held-out-environment
generalization over a total-parameter-matched dense model, and which routing choices explain the
gain or failure?

## Where the project is

The study is in Stage 0. The frozen 36-cell factorial has not started. Seven seed-0 shared-HPO
workers are revalidating the six frozen candidates per dataset under the single execution commit
`26ad7fa`; no common-commit result JSON is complete yet (`0/12`). The four RxRx1 workers have
finished their 30 training epochs and are in end-of-run evaluation. The three Camelyon17 workers
were at epochs 4/10, 2/10, and 1/10 at the latest evidence snapshot.

RxRx1 has an additional sanity gate because both 90-epoch DINOv2 probes remained unexpectedly
weak. A canonical WILDS ResNet-50 ERM reproduction is now training on the eighth GPU. It uses the
official RxRx1 transform and optimizer schedule and is restricted to validation and ID-test
evaluation; the OOD-test subset is not constructed or evaluated.

## Decision-grade findings

No comparative performance finding is decision-grade yet. In particular, there is not yet a
valid basis for selecting an RxRx1 or Camelyon17 recipe, comparing MoE with dense controls, or
making a mechanism claim.

The following protocol facts are verified:

- the formal Phase-A revalidation uses one tested code commit (`26ad7fa`);
- all Stage-0/1 selection remains on `ood_val`, with `test_evaluated=false` required;
- the earlier 12-result grid spans multiple commits and is diagnostic only;
- the RxRx1 canonical reproduction has an explicit validation-only split guard.

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

The canonical WILDS reproduction has begun epoch 0 and is healthy. Its first hundreds of batches
are implementation evidence only; they are too early for scientific interpretation.

## Current scientific interpretation

The low RxRx1 numbers should not yet be interpreted as evidence that DINOv2, sparse routing, or
batch-effect adaptation fails. The strongest current concern is a protocol mismatch: the project
loader uses ImageNet normalization with random resized cropping, whereas the canonical WILDS
recipe standardizes each image channel and uses rotations plus horizontal flips. That difference
can alter the biological signal before the architecture is tested.

Competing explanations remain live:

1. **Preprocessing mismatch:** the DINOv2 pipeline removes or distorts signal that the canonical
   RxRx1 pipeline preserves.
2. **Optimization mismatch:** the pretrained transformer needs a different schedule or adaptation
   budget than the current bounded probes.
3. **Representation mismatch:** natural-image pretraining may genuinely transfer poorly to this
   microscopy task.
4. **Task difficulty:** even the canonical WILDS reproduction may be weak in this exact environment,
   pointing to data/version or implementation issues rather than an MoE-specific problem.

The canonical ERM curve is the clean discriminator now. If it reaches the expected qualitative
learning regime while the DINOv2 probes remain near chance, the DINOv2 adaptation substrate needs
a bounded preprocessing/optimization correction before factorial comparison. If it is also weak,
the data version, labels, split handling, and official reproduction must be audited before any
architectural conclusion.

## Implications for experimental design

- Do not rank or replicate the current RxRx1 HPO cells until the sanity gate is resolved.
- Do not change factorial factors, levels, losses, or selection rules in response to this diagnostic.
- Camelyon17 may advance independently when its six valid seed-0 candidates complete; it need not
  wait for the RxRx1 diagnosis.
- Treat routing, decodability, and embedding analyses as mechanism evidence only after the matched
  predictive comparison is valid.

## Next decisions

1. Strictly validate and rank each dataset's six common-commit seed-0 results when complete.
2. Compare the canonical RxRx1 learning curve with the DINOv2 probes without accessing OOD test.
3. If the canonical run succeeds qualitatively, run only the predeclared bounded RxRx1 substrate
   correction needed to restore a credible dense baseline; otherwise audit data/version/splits.
4. Freeze shared recipes and calibration values before launching any Stage-1 factorial cells.

