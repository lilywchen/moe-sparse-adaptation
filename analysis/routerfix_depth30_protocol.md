# Corrected-router sparse-depth protocol

## Question

Can task-trained, function-preserving sparse upcycling improve supervised Cell-DINO adaptation
to RxRx1 held-out acquisition batches, and how much of the final transformer should be sparse?

The historical hard top-1 implementation renormalized one selected probability by itself. Its
forward value was one and its classification gradient to the router was effectively erased. The
new `selected_st` estimator keeps the forward value exactly one while retaining the selected
softmax probability's gradient. Every corrected run encodes the estimator in its run identity.

## First wave: eight GPUs, seed 0, 30 epochs

| Arm | Variant | Converted FFNs | Main role |
|---|---|---:|---|
| original | Cell-DINO | 0 | adaptation anchor |
| dense_last2 | dense-wide E8 | 10, 11 | equal-total-parameter control |
| learned_last1 | learned MoE E8 | 11 | minimal late sparse model |
| frozen_last1 | frozen-router MoE E8 | 11 | last-1 routing control |
| learned_last2 | learned MoE E8 | 10, 11 | primary corrected model |
| frozen_last2 | frozen-router MoE E8 | 10, 11 | primary routing control |
| learned_last4 | learned MoE E8 | 8–11 | moderate sparse depth |
| learned_all12 | learned MoE E8 | 0–11 | full-upcycling endpoint |

All arms use the native five-channel Cell-DINO input, full end-to-end fine-tuning, seed 0,
identical data order, ERM, 30 epochs, milestones at 5/10/20/30, and no OOD-test evaluation.

## Readout order

1. **Correctness:** each learned top-1 arm must report a finite, nonzero classification-only
   router gradient before its first update. Trainable router parameters must move from their
   initial values; frozen-router parameters must not.
2. **Adaptive-routing gain:** learned minus frozen at last 1 and last 2 FFNs.
3. **Conditional-capacity gain:** learned last-2 MoE minus dense last-2 at matched total capacity.
4. **Capacity gain:** each learned MoE minus original Cell-DINO.
5. **Depth response:** original versus last 1, last 2, last 4, and all 12.
6. **Mechanism:** randomized-route reliance, expert usage/entropy, and OOD worst-experiment
   movement. Mean OOD improvement without route reliance is not evidence for adaptive routing.

Epoch 5 and 10 are trajectory diagnostics, not negative stopping points. Epoch 20 can identify a
clear failure or leader; epoch 30 is the first performance shortlist point. A practical promotion
signal is at least +0.5 percentage points on OOD validation versus the relevant frozen/dense
control, no meaningful worst-experiment regression, and nontrivial randomized-route reliance.
These are resource-allocation thresholds, not confirmatory claims.

## Second wave

Choose the winning block set without changing its recipe. Fill eight GPUs with the exact
original/dense/learned/frozen quartet at seeds 1 and 2. Together with seed 0, this gives three
paired seeds for the primary routing, conditional-capacity, and capacity contrasts within a
second 30-epoch wall-clock window.

Example for the last two blocks:

```bash
python scripts/sweep_rxrx1_routerfix_depth30.py \
  --phase confirm --confirm-blocks 10,11 --shard-index 0 --num-shards 4
```

Use shard indices 0, 1, 2, and 3 once each across the four two-GPU containers.
