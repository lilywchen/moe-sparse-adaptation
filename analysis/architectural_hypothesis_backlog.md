# RxRx1 Cell-DINO architectural hypothesis backlog

Status: predeclared exploratory backlog, 2026-08-01 23:39 EDT. This registry extends the active
factorial60 queue; it does not interrupt healthy workers and does not license OOD-test access.
The active factorial60 campaign already has 14 tested runnable arms, so its immediate refill queue
exceeds the 12-arm minimum. The rows below are the next bounded architectural questions to make
runnable through isolated implementation, regression testing, smoke tests, and exact matched
controls. A selected seed-0 winner remains exploratory until its configuration is locked and
replicated unchanged at fresh seeds.

## Evidence-driven ordering

The current signal is specific: early token-cosine routing leads the new epoch-10 screen on mean
OOD validation, while early image-linear route pressure leads the first epoch-30 screen but still
has weak worst-experiment accuracy. The validated hypothesis90 campaign showed a small canonical
MoE gain that peaked at epochs 30--60 and shrank by epoch 90. This makes preservation, router
geometry, tail safety, and conflict-localized placement sharper questions than another broad
optimizer sweep.

Primary literature used to form the backlog:

- [Sparse Upcycling](https://arxiv.org/abs/2212.05055) initializes vision MoE experts from a dense
  checkpoint and motivates function-preserving continuation rather than destroying a competent
  representation at the start of task adaptation.
- [ST-MoE](https://arxiv.org/abs/2202.08906) motivates router-logit stabilization and separating
  router stability from load-balancing pressure.
- [Expert Choice Routing](https://proceedings.neurips.cc/paper_files/paper/2022/hash/2f00ecd787b432c1d36f3de9800728eb-Abstract-Conference.html)
  fixes an expert's token budget and motivates a direct token-choice versus expert-choice test.
- [Routers in Vision MoE](https://arxiv.org/abs/2401.15969) reports strong expert-choice and soft
  routing in vision under fixed compute, motivating both as separate fairness estimands.
- [ViMoE](https://arxiv.org/abs/2410.15732) motivates a shared dense expert and routing-behavior
  diagnostics because some vision layers do not learn reliable specialization.
- [GMoE for domain generalization](https://arxiv.org/abs/2206.04046) motivates combining sparse
  architecture with, rather than substituting it for, domain-generalization objectives.
- [Fish gradient matching](https://openreview.net/forum?id=vDwBW49HmO) motivates measuring and then
  intervening on inter-experiment gradient alignment.
- [AGRO](https://iclr.cc/virtual/2023/poster/10942) motivates tail-aware objectives when the
  relevant failure groups are not fully captured by a single predefined partition.

## Common protocol

Unless a row says otherwise: native CP5, seed 0, fixed factorial60 data order, AdamW base
`lr=1e-4`, `wd=0.05`, batch 64, OOD-validation selection, OOD test sealed, checkpoints at
10/30/60, and successive halving at epochs 10 and 30. Every sparse arm gets the stated same-seed,
same-placement, same-objective dense control. Exact-total-parameter, active-compute, and original-
budget claims are reported separately. Target continuation to epoch 60 requires a competitive
trajectory or the predeclared mechanism signature. Advancement to epoch 90 or seeds 1/2 requires
at least +5 absolute OOD-validation points over the proper dense control with no more than two ID
points lost, or a smaller mean gain with a consistent predeclared worst-experiment improvement.

## Ranked bounded questions

| Rank | Predeclared arm and exact comparator | Predicted signature | Alternative explanation and falsifier | Fairness / budget / status |
|---:|---|---|---|---|
| 1 | Upcycle the early-block dense checkpoint into 8 cloned token-cosine top-1 experts; compare with continued early dense-wide from the identical checkpoint. | Preserves the dense epoch-10/30 OOD score initially, then opens a MoE gain without an ID drop. | Extra continuation alone explains the gain; falsified if the paired dense trajectory matches it or routing stays diffuse. | Exact-total-parameter matched; 10/30/60; implement checkpoint conversion and resume tests. |
| 2 | Rank 1 plus 0.01 relative expert symmetry-breaking noise; same dense control and a zero-noise upcycling control. | Earlier expert differentiation with no abrupt representation loss. | Noise is ordinary regularization; falsified if a noise-matched dense branch performs equally or routes remain unstable/dead. | Exact-total-parameter; 10/30/60; bounded noise pair only. |
| 3 | Rank 1 with router-only warmup for 1 epoch, then expert unfreezing; compare with simultaneous unfreezing. | Route entropy and augmentation stability settle before experts diverge; OOD lead appears by epoch 30. | Delay merely lowers effective training; falsified by no mechanism change or weaker matched OOD/ID. | Exact-total-parameter; 10/30/60; implement staged optimizer groups. |
| 4 | Rank 1 with experts-only warmup and frozen uniform router for 1 epoch, then router unfreezing. | Experts leave clone symmetry before hard routing can starve them. | Delay or random partitioning explains the effect; falsified if expert usage remains symmetric/diffuse and OOD does not improve. | Exact-total-parameter; 10/30/60; complementary staged control to rank 3. |
| 5 | Early token-cosine top-1 with router z-loss crossed at `{0, 1e-3}` while holding balance strength fixed; paired dense control unchanged. | Smaller router-logit excursions and more stable routes without forcing uniform usage. | Generic regularization; falsified if logits were already stable or dense-equivalent OOD remains. | Exact-total-parameter; four-arm budget including controls; implement stable log-sum-exp metric. |
| 6 | Early token-cosine top-1 with load-balance coefficient `{0, 0.01, current}`. | Low/zero balance allows useful experiment-agnostic specialization and improves mean OOD without dead experts. | Apparent gain is expert collapse; falsified by dead-expert rate, unstable augment routes, or tail loss. | Exact-total-parameter; three sparse arms sharing one dense control; epoch 10/30 screen. |
| 7 | Expert-choice early routing with fixed per-expert capacity versus token-choice early token-cosine top-1 and exact dense. | Better expert utilization and worst-experiment accuracy at similar active compute. | More tokens or variable fan-out buys extra compute; falsified under measured active-FLOP matching. | Active-compute matched primary, exact-total secondary; implement deterministic bucket/capacity accounting. |
| 8 | Soft early MoE with 8 experts and fixed slot budget versus expert-choice, token-choice, and dense. | Smoother credit assignment improves OOD and route stability before hardening. | It is simply more active compute; falsified by active-FLOP-matched dense and hard-router controls. | Active-compute matched only; do not pool with hard sparse efficacy. |
| 9 | Add one shared dense expert plus 7 routed experts at early placement; compare with an 8-expert sparse arm and parameter-matched dense. | Shared expert preserves common morphology while routed experts absorb experiment-specific residuals; tail improves. | Total active compute explains result; falsified after active-compute and total-parameter controls. | Two separately labeled estimands; 10/30/60; implement shared-path accounting. |
| 10 | Early well-level routing: pool replicate images from the same well only, never experiment identity; compare with image and token routing. | Routes stabilize across views and specialize by morphology rather than pixels. | Replicate aggregation leaks a label proxy; falsified by label-shuffled well grouping and no augmentation-stability gain. | Exact-total-parameter; requires audited well-key sampler without held-out metadata input. |
| 11 | Augmentation-consistency regularization on early token-cosine route assignments versus an equal-weight feature-consistency dense control. | Higher route stability and OOD with preserved ID; reduced randomized-router sensitivity. | Any consistency loss helps; falsified if dense control matches or route stability changes without performance. | Exact objective/parameter pair; two augment views; bounded coefficient `{0.01,0.1}` only. |
| 12 | Preservation penalty to the starting Cell-DINO features on early MoE versus the same penalty on early dense-wide. | Shrinks ID-to-OOD forgetting while retaining a conditional gain near epochs 30--60. | Feature distillation alone explains gain; falsified if dense control matches or experts never specialize. | Exact-total-parameter and objective matched; coefficient chosen from one two-point screen. |
| 13 | Measure per-layer inter-experiment gradient cosine at original checkpoints 0/10/30; place MoE only at the strongest reproducible conflict peak and compare with equal-width dense at that layer. | Sparse benefit concentrates where source-experiment gradients disagree, with improved post-intervention alignment. | Layer depth alone explains it; falsified by a nonconflict layer placebo or no alignment/performance change. | Diagnostic first, then exact-total-parameter pair; no held-out gradients. |
| 14 | Two sparse blocks only at the top two measured conflict peaks versus two equally widened dense blocks. | Larger OOD gain than either single block without proportional ID loss. | Extra capacity/optimization explains it; falsified by dense pair and no superadditive mechanism signature. | Exact-total-parameter; epoch 10 prune unless delayed emergence is supported by gradient diagnostics. |
| 15 | Freeze shared blocks with aligned source gradients and adapt only conflict-localized sparse blocks; compare with conflict-localized dense adaptation. | Prevents global overfitting while allowing conditional conflict resolution. | Reduced trainable parameters regularize both models; falsified if matched dense is equal or better. | Trainable-parameter and total-parameter reported separately; 10/30/60. |
| 16 | Early route-pressure MoE crossed with GroupDRO over training experiments; compare with early dense-wide GroupDRO and MoE ERM. | Retains the mean OOD signal while reversing the worst-experiment penalty. | GroupDRO alone explains it; falsified if dense GroupDRO matches or mean collapses. | Exact objective/parameter matched; tail-safe lane; 10/30/60. |
| 17 | Early route-pressure MoE with a soft worst-experiment constraint activated after epoch 10; compare with the same constraint on dense. | Mean OOD remains competitive and tail begins improving after activation. | Later objective change is generic fine-tuning; falsified by dense control or unstable tradeoff across milestones. | Exact objective pair; one predeclared constraint strength; no sweep over activation epoch. |
| 18 | Learned adversarial group discovery on training features followed by MoE/dense robust objectives, with discovery frozen before comparison. | Finds error-prone slices that correlate with weak validation environments and improves tail OOD. | Slice model exploits class imbalance; falsified by label-stratified checks and dense equivalence. | Diagnostic then exact pair; groups use train only; implementation-required. |
| 19 | Route frozen-random, route learned, and route permuted-at-eval controls for the early token-cosine survivor. | A useful router shows higher stable mutual information with morphology and a measurable eval-time reliance gap. | Experts act as an ensemble independent of routing; falsified if random/permuted routing matches. | Mechanism analysis, not efficacy; reuse matched saved checkpoints where possible. |
| 20 | Vary experts `{4,8,16}` while holding total parameter budget fixed by shrinking expert width; pair each with exact-width dense. | If specialization rather than raw capacity matters, an intermediate expert count wins with nondead usage. | Parameterization artifacts explain it; falsified by monotonic dense-equivalent scaling or dead experts. | Original-total-budget primary; 10/30 screen; implement width solver and exact counts. |
| 21 | Capacity factor `{1.0,1.25}` for expert-choice routing with explicit dropped-token accounting. | Modest slack improves tail examples without changing typical active compute materially. | Extra compute explains it; falsified under exact processed-token/FLOP reporting. | Active-compute matched; two-point screen only. |
| 22 | Router temperature schedule: fixed `1.0` versus anneal `2.0→0.7` for early token-cosine. | Soft early credit assignment followed by sharper specialization improves epoch-30 OOD and route stability. | Generic learning-rate effect; falsified if entropy changes without conditional gain. | Exact-total-parameter; share dense and fixed-temperature comparator. |
| 23 | Layerwise learning-rate decay preserving early shared features, crossed only with the best upcycling arm and its dense control. | Reduces late forgetting so the epoch-60 gain does not collapse by epoch 90. | Ordinary optimization rescues both equally; falsified by matched dense or no persistence. | Exact pair; licensed only after a rank 1--12 arm survives epoch 30. |
| 24 | Locked winner ablation: learned router versus nearest-prototype cosine router computed from train-only features. | Prototype routing is more stable across experiments if learned routing overfits source batches. | Prototypes encode labels directly; falsified with class-agnostic morphology prototypes and dense controls. | Mechanism/exploratory; no held-out experiment identity; implement only after a reproducible learned-router lead. |

## Required measurements for every new MoE arm

In addition to train/ID/OOD-validation/worst-experiment accuracy: expert usage and dead-expert
rate, normalized routing entropy, router-logit range, per-experiment usage, route stability under
augmentation, randomized/permuted-router reliance, active token-expert assignments, dropped-token
rate, measured active FLOPs, and exact total/trainable/router parameters. A mechanism signature can
keep a delayed-emergence arm alive only when it was predeclared above; it cannot replace the final
matched performance comparison.

## Immediate refill policy

The 14 existing factorial60 cells remain the only currently tested runnable refill queue. No row
above may displace a healthy worker. As leases turn over, factorial60's exact early-canonical dense
comparator stays highest priority until its matched milestone exists. In parallel, implementation
work starts with ranks 1, 5--7, 9, 11--13, 16, 19, and 20 so at least 12 next-family arms can pass
tests and dry runs before factorial60 exhausts its queue.
