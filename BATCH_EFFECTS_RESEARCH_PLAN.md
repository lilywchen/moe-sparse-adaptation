# RxRx1 batch effects: falsification and transport-MoE plan

## The claim being tested

A batch effect is not any representation signal that predicts experiment identity. It is the
experiment-dependent component that degrades perturbation recognition in a new experiment. The
working decomposition is

\[
h_{e,y,i}=\mu_y+b_e+q_{e,y}+\epsilon_{e,y,i},
\]

where biology is \(\mu_y\), additive shift is \(b_e\), interaction is \(q_{e,y}\), and within
experiment×perturbation/site noise is \(\epsilon\). All comparisons are within cell type unless
cell type is explicitly modeled; experiment and cell line are nested in RxRx1 and must not be
confounded.

## Falsification ladder

| Hypothesis | Model of the representation | Data test | Architectural consequence |
|---|---|---|---|
| H0 | biology + noise | raw leave-one-experiment-out retrieval is already high | no correction |
| H1 | additive shift | centering closes the generalization gap and interaction energy is small | batch translation only |
| H2 | diagonal affine shift | AdaBN materially improves over centering | AdaBN |
| H3 | low-rank interaction | experiment×perturbation residual has low effective rank | one low-rank residual |
| H4 | shared correction family | a few operator components explain source experiments and are predictable for held-out experiments from unlabeled moments | set-conditioned transport MoE |
| H5 | idiosyncratic interaction | operator family is high-rank or held-out coefficients are unpredictable | do not deploy MoE correction |

The H4 method is

\[
z'=\operatorname{AdaBN}(z;C_e)+\sum_k \alpha_k(C_e)U_kV_k^\top
\operatorname{AdaBN}(z;C_e).
\]

The router consumes a permutation-invariant descriptor of an unlabeled support set \(C_e\), not
an experiment ID. The residual operators are shared and zero-output initialized. Applying an
operator to \(z\) already makes the shift phenotype-dependent; a second per-cell router is an
optional discovery ablation, not the primary bet.

## The core training idea

RxRx1 repeats perturbations across experiments. Each transport-training minibatch samples one
cell type, two experiments, and the same 32 perturbations from both. This makes the two empirical
distributions composition matched: set-level differences identify acquisition rather than a
different phenotype mixture. A supervised contrastive term pulls each matched perturbation pair
together while all other perturbations remain negatives. This is the Harmony ingredient—align
biology across batches without erasing biological neighborhoods—while AdaBN supplies the affine
anchor.

The `PairedERM` arm separates the paired sampler from its alignment loss. `HarmonyDG` uses the
pairing and alignment only and consumes zero target information at inference. `TransportMoE`
adds unlabeled target context. This distinguishes a training-data effect from a correction-model
effect.

## Difficulty is a distribution, not one number

Every experiment receives a profile containing:

- raw, centered, and AdaBN-corrected perturbation retrieval;
- predictive harm and correctable gain;
- additive and interaction energies and their ratio;
- effective interaction rank;
- within experiment×perturbation site/image noise;
- source portability and held-out correction-coefficient predictability.

The report includes the distribution and hardest tail of these quantities, plus rank correlations
between predictive harm and nuisance size, interaction, rank, and noise. Perturbation difficulty
is estimated separately from its error rate across experiments.

Uncertainty uses a crossed experiment×perturbation bootstrap stratified by cell type. Images are
not treated as independent replicates. Confirmatory method differences use a paired crossed
seed×experiment bootstrap. Target-context curves use eight repeated support/query draws at 8, 16,
32, and 64 unlabeled images; support observations are removed from scoring.

## Ambition and information ladder

| Level | Method | Learned during training | Target-time information |
|---|---|---|---|
| L0 | grouped ERM | classifier/backbone | none |
| L1 | HarmonyDG | composition-matched cross-batch invariance | none |
| L2 | AdaBN | backbone plus affine parameters | unlabeled support moments |
| L3 | TransportMoE | shared low-rank transport family and set router | the same unlabeled support moments |
| L4 | oracle correction ceiling | source operator family | labeled matched target perturbations; analysis only |

The preferred result is L1 matching AdaBN. If it does not, the next claim is L3 matching or
beating AdaBN with fewer or equal unlabeled support images. An oracle result is never a deployable
win.

## Twelve-hour compute plan

Eight H100s run 36 predeclared jobs (about 85 GPU-hours plus final geometry extraction):

1. Discovery: 12 methods/ablations × one seed × 12 epochs; OOD validation only.
2. Replication: four core methods × three fresh seeds × 30 epochs; OOD validation only.
3. Confirmation: the same four methods × three fresh seeds × 100 epochs; test is unsealed for
   these predeclared arms only.
4. Frozen-feature audit and automatic aggregation run after every shard reaches a terminal state.

The four core methods are grouped ERM, HarmonyDG, AdaBN, and TransportMoE. Longest jobs launch
first so three 100-epoch runs balance over the two GPUs in each container. One process is allowed
per GPU. All code, caches, logs, checkpoints, manifests, failures, and reports live in SciServer
persistent storage.

## Paper anchors and interpretation

The paper reports 75.1% for its batch-separated baseline and 87.1% for AdaBN. Its training recipe
uses DenseNet-161, six-channel 512×512 images, batch size 512, 100 epochs, and eight A100 GPUs. The
local benchmark uses Cell-DINO at 128×128 and therefore reports the combined local validation
(four experiments) plus test (14 experiments) as a split-compatible reasonableness anchor, not an
exact reproduction. Model selection never uses the 14-experiment test partition.

Batch decodability is diagnostic only. A method succeeds by improving unseen-experiment
perturbation accuracy and the worst-experiment tail while preserving biology. A shared-operator
claim additionally requires low family rank, positive held-out descriptor-to-operator prediction,
multiple effective experts, and nonzero route/correction reliance.

## One-line operation

Run one shard per 2×H100 container:

```bash
python scripts/sweep_rxrx1_batch_correctors.py --shard-index <0..3> --num-shards 4 --gpus 0,1 --max-concurrent 2 --wait-for-global
```

Use `--wait-for-global` only on shard 0. It performs the final test-blind feature audit and writes
`aggregate.json`, `REPORT.md`, and `hypothesis_audit/batch_hypotheses.md` when all runs finish or
fail explicitly.
