# Predeclared study plan

## Scientific question

When adapting a pretrained vision encoder to scientific images collected across acquisition
environments, when does sparse conditional capacity improve held-out-environment generalization
relative to an equally large dense adaptation?

The paper is not a leaderboard claim and does not claim that MoE is a universal batch-effect
correction. It asks whether conditional computation is useful in this setting, identifies the
design choices that govern any gain, and tests what the learned routing relies on.

## Three questions

1. **Does conditionality help?** Compare learned top-1 MoE with a depth-matched dense-wide FFN at
   the same total parameter budget, seed, data order, optimizer, and training objective.
2. **Which design choices determine the gain?** Estimate main effects and two-way interactions for
   placement × routing unit × router geometry × training pressure, rather than declaring one
   isolated winning recipe.
3. **What mechanism accompanies success or failure?** Relate conditional gain to expert usage,
   route dependence, acquisition/label information in routes, and acquisition/label decodability
   in final embeddings.

## Fixed substrate and datasets

- DINOv2 ViT-S/14, full supervised fine-tuning.
- RxRx1: microscopy perturbation classification; acquisition environment = experiment.
- Camelyon17: histopathology tumor classification; acquisition environment = hospital.
- Exactly one FFN block is converted in every capacity intervention.
- Eight experts, top-1 routing, original expert width.

These two datasets support a paired cross-modality characterization, not a claim over all
scientific imaging. Conclusions must report both agreement and disagreement between them.

## Primary factorial

| Factor | Levels | Scientific role |
|---|---|---|
| Placement | early / middle / late | where conditional capacity enters the representation hierarchy |
| Routing unit | image / token | whether decisions use global acquisition context or local morphology |
| Geometry | linear / cosine | whether feature magnitude may influence routing |
| Pressure | canonical / route / output | unconstrained routing, within-environment route balance, or final-feature invariance |

The 36 MoE cells are all run on both datasets with seeds 0, 1, and 2. This is a factorial, not a
sequential search: loss pressure is not tuned after selecting an architecture.

## Training pressures

- `canonical`: global Switch-style load balancing plus router z-loss.
- `route`: the same balancing loss computed within each acquisition environment and averaged.
  This equalizes marginal expert usage within environments; it does not claim route invariance.
- `output`: canonical load balancing plus a gradient-reversal adversary predicting the training
  environment from the final embedding.

Dense-wide controls use `canonical` and `output`. The output arm is compared DANN-for-DANN.
Route-pressure MoE uses the canonical dense comparator because route-level balance is undefined
for a dense model.

## Parameter and optimization fairness

- The fixed budget is total model parameters, not active parameters.
- Dense hidden width is the integer width closest to the corresponding full MoE block budget,
  including router and replicated expert biases. The residual delta must be below 0.1% and is
  logged per run; no dormant padding parameters are allowed.
- Dense upcycling uses exact function-preserving Net2Wider initialization with unequal outgoing
  splits that sum to one. MoE experts are exact copies of the pretrained FFN.
- Linear and cosine routers have the same parameterization: one E × d matrix and one scalar
  temperature.
- Output-invariance runs include the same adversary architecture on both sides; training-total and
  inference-total parameters are reported separately.
- All factors share the same augmentation, sampler, epochs, optimizer family, LR schedule, layer-
  wise decay, batch size, and seed.

## Hyperparameter protocol

Before the factorial, tune the shared dense full-fine-tuning recipe on inner OOD validation only.
Then tune one router temperature per dataset × geometry, one load-balancing weight per dataset,
and one output-invariance weight per dataset. Do not tune per cell. Freeze these values before
running the 36-cell grid and record all candidates, selection metrics, and compute.

## Outcomes and contrasts

Primary outcome: OOD-validation accuracy in Stage 1/2 and held-out OOD-test accuracy in Stage 3.
Always report ID accuracy, worst-environment accuracy, and the ID-to-OOD degradation gap.

Primary contrast: `conditional_gain = MoE − matched dense-wide`.

Secondary controls include learned versus frozen router, original-width DINOv2, and learned versus
randomized routes with expert weights fixed.

## Mechanism analyses

1. Expert usage, entropy, load imbalance, and dead experts by layer and environment.
2. Route mutual information / cross-validated decodability for environment and biological label.
3. Final-embedding environment and label decodability with identical linear probes.
4. Counterfactual randomized routing and route reliance.
5. Associations between conditional gain and these quantities across cells.
6. UMAP only as a qualitative appendix figure; no inference is based on a 2-D projection.

## Stage gate

- **Stage 0:** protocol tests, dry run, shared HPO, and budget audit.
- **Stage 1:** complete 36-cell grid with three seeds, OOD validation only.
- **Stage 2:** predeclared frozen-router and mechanism controls on finalists and representative
  failures selected without OOD test access.
- **Stage 3:** fresh seeds on the fixed confirmatory set; only here may OOD test be evaluated.

No headline claim is allowed if the budget audit fails, the OOD test influenced selection, or a
reported cell lacks its paired dense control at the same seed.
