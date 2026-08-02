# RxRx1 Cell-DINO architectural hypothesis backlog

Status: predeclared exploratory backlog, updated 2026-08-02 14:20 EDT. This registry extends the active
factorial60 queue; it does not interrupt healthy workers and does not license OOD-test access.
The active campaign has fifteen tested runnable arms after five released GPUs were assigned
canonical/route E32, canonical/route E64, and broader route-E4 tail-safe. An eight-cell extreme
expert-bank by pressure by auxiliary-loss bracket passed isolated SciServer tests and an exact dry
run, restoring the queue above the 12-arm minimum. The
rows below are the next bounded architectural questions to make
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
| 25 | Extend the low-temperature plus-z-loss expert-bank bracket to canonical and route-pressure E32, sharing the completed E8 anchor. | If useful specialization continues with bank size, E32 improves mean OOD with nondead, stable expert use; if E16 is already sufficient, the curve saturates. | Raw parameter count or router fragmentation explains any movement; falsified by an E32 tie/loss versus E8/E16, dead experts, or unstable routes. | Active-compute matched only, not exact-total or original-budget matched; seed 0; 10/30/60; two-arm bounded extension; total/active FFN-plus-router parameters predeclared as 59,043,060/1,193,857. |
| 26 | Extreme canonical/route E64 overfragmentation bracket at the same low-temperature plus-z-loss setting, sharing E8 and the E16/E32 trajectory. | If conditional capacity remains useful, E64 improves OOD without dead experts; otherwise route fragmentation or excess inactive parameters causes saturation or degradation. | A gain is raw capacity rather than routing; falsified by no E64 improvement over E16/E32, dead experts, unstable augmentation routes, or worse tail. | Active-compute matched only; seed 0; 10/30/60; two-arm hard upper bracket; total/active FFN-plus-router parameters predeclared as 96,865,524/1,206,145. |
| 27 | Cross E32/E64 and canonical/route pressure with tail-safe auxiliary loss `(balance,z)=(0.01,0.01)` versus no auxiliary loss `(0,0)`, holding temperature 0.03 and data order fixed. | Auxiliary pressure prevents dead or highly imbalanced experts at extreme bank sizes and improves worst-experiment accuracy without erasing mean OOD. | Bank size is irrelevant or auxiliaries only add generic regularization; falsified by no usage/entropy interaction and no OOD/tail improvement over no-aux. | Active-compute matched only; eight seed-0 arms, 10/30/60 milestones; new W&B group/HF folder; never pool with exact-total efficacy. |

## Required measurements for every new MoE arm

In addition to train/ID/OOD-validation/worst-experiment accuracy: expert usage and dead-expert
rate, normalized routing entropy, router-logit range, per-experiment usage, route stability under
augmentation, randomized/permuted-router reliance, active token-expert assignments, dropped-token
rate, measured active FLOPs, and exact total/trainable/router parameters. A mechanism signature can
keep a delayed-emergence arm alive only when it was predeclared above; it cannot replace the final
matched performance comparison.

## Immediate refill policy

Fifteen nonduplicate arms are tested and dry-run ready after E32/E64 pairs and one broader cell
entered service. The remaining broader and extreme-auxiliary brackets are remotely licensed; none
displaced a healthy worker:

1. seven remaining expert-count architecture arms crossing route/canonical pressure, 4/16 experts, and
   tail-safe versus zero auxiliary pressure; these are active-compute matched and must not be
   described as exact-total-parameter or original-budget matched; and
2. eight E32/E64-by-pressure-by-auxiliary cells testing whether balance/z-loss prevents extreme-bank
   fragmentation.

Broader canonical-E4 tail-safe has first refill priority, followed by canonical-E4 zero-auxiliary
and the other five broader cells, then the eight extreme-auxiliary cells. Expert-count arms answer the
next architectural question rather than retuning the selected pair. All arms inherit the sealed
OOD test, 10/30/60 checkpoints, exact data ordering within each pair, declared W&B groups and HF
campaign folders, and milestone validation before scientific consumption. The 27 ranked questions
above remain the hypothesis backlog.
