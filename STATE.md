# Current scientific state

Last updated: 2026-08-09 EDT

This is the compact source of truth for the current question, evidence, interpretation, and active
experiment. `PROGRESS.md` remains the chronological ledger; older exploratory analyses remain in
`analysis/`.

## 2026-08-09 steward result — fresh seed 1 replicates the shared-residual advantage

Campaign `shared_confirm30_20260809` now has all four terminal seed-1 rows. OOD validation is the
decision metric; OOD test remains a descriptive fixed-arm readout.

| Seed-1 arm | Train | ID | OOD val | OOD test | Worst test batch | Route reliance |
|---|---:|---:|---:|---:|---:|---:|
| Original Cell-DINO | 100.000% | 52.581% | 20.459% | 36.454% | 6.639% | — |
| Dense E4, late 2 | 100.000% | 55.188% | 21.839% | **38.981%** | 7.869% | — |
| Replacement E4/top-2, late 2 | 100.000% | 54.223% | 20.926% | 37.715% | 6.844% | 0.0507 |
| Shared residual E3/top-1, late 2 | 100.000% | **55.612%** | **22.346%** | **38.758%** | **8.811%** | 0.0466 |

The seed-1 validation deltas are `+0.507` points versus dense, `+1.421` versus matched replacement
MoE, and `+1.887` versus original Cell-DINO. Relative to dense, shared is `-0.224` on descriptive
mean test, `+0.943` on worst test batch, and `+0.424` on ID. Relative to replacement, its test
delta is `+1.043`, worst-batch delta is `+1.967`, and ID delta is `+1.389` points.
The validation direction now agrees with seed 0: shared beat dense by `+0.728` and replacement by
`+0.474` in the previous wave. Seed 2 remains required before treating the effect size as stable.

This is not an added-compute artifact. The audit reports replacement at `29,494,645` total /
`4,729,346` active FFN parameters and shared residual at `29,493,877` total / `4,728,578` active—
only 768 parameters apart in each count. Dense has the same total capacity (`29,493,881`) but
activates `9,454,854` FFN parameters, almost exactly twice shared residual. Thus shared beats dense
on validation and tail accuracy at roughly half the active FFN compute, while remaining within
`0.224` test points. Both routers are consequential by the predeclared reliance gate, but shared
wins despite slightly *lower* route reliance than replacement. The evidence favors preserving the
pretrained dense FFN and routing a residual correction, not simply stronger routing or more active
capacity.

Both sweep controllers remain healthy and all four seed-2 arms are training. No follow-up
architecture wave should displace this replication.

## 2026-08-09 steward pass — confirmation wave launched; batch diagnostics queued

Live reconstruction found both scoped bottom SciServer containers running but their four H100s
idle. The persistent checkout was at `90a4e80`; GitHub `main` is newer (`023bd18`) and contains the
checkpoint-only mechanism audit. No new scientific number was inferred from GPU idleness.

The next performance wave is now frozen as `shared_confirm30_20260809`: two fresh seeds (`1,2`)
times four arms, all at 30 epochs and blocks 10–11 under the same Cell-DINO/RxRx1 protocol.

Launch is verified at immutable commit `9365406` after `289` tests passed on SciServer. Shard 0 is
running on container `2862` and shard 1 on container `2859`, each with two active H100 processes.
The first four distinct arms started without an immediate error; the remaining four seed-2 runs
are held by the two live sweep controllers and will start as those GPU slots free. At launch, all
four GPUs showed active training processes and the four fresh logs contained no traceback. Local
artifacts are authoritative because W&B and Hugging Face credentials are absent in the containers;
the result root is
`substrate_rxrx1/cell_dino_cp5/shared_confirm30_20260809`.

| Arm per seed | Role |
|---|---|
| `original` | Untouched Cell-DINO adaptation baseline |
| `dense_E4_late2` | Ordinary equal-total dense expansion |
| `replace_E4k2_late2` | Traditional replacement MoE control |
| `shared_E3k1_late2` | Leading shared residual MoE |

This is an unusually clean comparison. Dense E4, replacement E4/top-2, and shared E3/top-1 each
allocate approximately four FFN banks at the two converted blocks. Replacement and shared each
activate two FFN banks; dense activates all four. Thus total capacity is matched, replacement and
shared active compute are matched, and shared uses roughly half the active FFN compute of dense.
The primary validation contrasts isolate allocation and conditional routing from generic total
capacity while also testing sparse versus dense activation. The primary estimands are
`shared-dense` and `shared-replacement` per seed; OOD validation decides and the fixed-arm OOD test
readout remains descriptive.

The mechanism audit now includes a correction-off counterfactual for shared residual MoE. At
inference, the same checkpoint can be evaluated with all routed residual corrections disabled.
Together with randomized routing, this separates three questions:

1. Does the residual branch contribute at all (`full - shared-only`)?
2. Does the learned route matter (`full - randomized-route`)?
3. Does shared sparse allocation beat equally active dense/replacement capacity across seeds?

A checkpoint-only batch/embedding analysis is also predeclared. It uses the same perturbations in
every experiment within each cell line, preventing label composition from being mistaken for
batch shift. OOD severity is measured independently in frozen pretrained Cell-DINO space as the
distance from a held-out experiment's class-residual centroid to its nearest training-experiment
centroid, normalized by within-cell train distances. The report will join per-experiment
accuracy/confidence/error overlap with class-versus-batch variance, cross-batch perturbation
retrieval, CKA/drift from pretraining, routing-distribution shift, and severity–accuracy slopes.
With only a few held-out experiments, correlations are descriptive diagnostics rather than
high-powered estimates.

The larger-data scaling branch remains preparation-only. The verified SciServer inventory has
native RxRx1 but no validated JUMP-CP/Cell Painting Gallery corpus with a declared supervised
label space and held-out-batch split. Duplicating or resampling RxRx1 would not test data scaling.
No large-corpus training should launch until the task, channels, controls, batch metadata,
license, storage, and split are concrete.

## 2026-08-08 resynchronization — two new waves

The state above was written when the shared/residual wave had only reached epochs 1–2. It is now
complete. A second, independently implemented frontier-MoE wave has also returned 7 of 8 planned
rows. The first wave answers the immediate performance question; the second distinguishes a
working conditional-routing mechanism from a mechanism that actually improves mean OOD transfer.

### Completed shared/residual performance wave

Campaign: `shared_residual_performance30_20260807` at code commit `d7fad7a`. All rows are seed-0,
30-epoch terminal readouts under the same Cell-DINO/RxRx1 protocol.

| Arm | OOD val | OOD test | Worst test batch | ID |
|---|---:|---:|---:|---:|
| `replace_E4k2_late2` | 21.768% | 37.634% | 8.361% | 53.716% |
| `shared_E3k1_late2` | **22.242%** | **38.877%** | 8.361% | **55.380%** |
| `shared_E3k2_late2` | 21.718% | 38.415% | 8.074% | 53.907% |
| `shared_E7k1_late2` | 21.240% | 37.143% | 8.730% | 53.260% |
| `shared_E3k1_late4` | 20.844% | 37.256% | 7.746% | 53.905% |
| `shared_E3k2_late4` | 20.854% | 36.969% | 8.852% | 53.536% |
| `shared_E3k1_xbatch` | 18.642% | 33.305% | 6.967% | 47.882% |
| `shared_E3k1_mixstyle` | 19.972% | 35.856% | 6.189% | 51.916% |

What changed:

- Keeping the pretrained dense FFN and adding a sparse residual correction is the first clean
  positive result of this phase. `shared_E3k1_late2` beats the current-code replacement reference
  by `+0.474` validation and `+1.243` test points with equal worst-batch accuracy.
- It also edges the earlier dense-expansion reference by `+0.728` validation and `+0.146` test
  points, while its worst test batch is `+0.410` points higher. This is promising single-seed
  performance evidence, not a replicated result.
- The useful allocation is specifically *three experts, top-1, and two late blocks*. Top-2,
  seven experts, and additional depth all reduce mean OOD performance. Deeper top-2 is the only
  variation that improves the tail, at a substantial mean cost.
- The two batch-robustness additions are negative in this implementation: cross-experiment
  consistency and MixStyle both reduce mean, test, and tail accuracy. They are not the next
  performance lever.
- No arm reaches the aspirational `30%` validation / `40%` test region. The exact endpoint has
  moved up modestly, not decisively.

### Frontier-MoE mechanism wave

Campaign: `frontier_moe30_20260807`, code commit `90a4e80` (the newer `main`). It intentionally
tests stronger conditional mechanisms: ground-truth-indexed oracle ceilings, conditional-statistic
and low-rank variants, a soft-routing E8 model, a GroupDRO control, and BTX specialists. Its
external comparison numbers come from a prior commit, so within-wave comparisons are safer than
cross-wave rankings.

- The two oracle ceilings do **not** clear dense validation: cell-type oracle is `20.276%` OOD
  validation (`-1.238` points versus the earlier dense reference) and environment oracle is
  `19.200%`. Thus, even explicitly indexed expert paths did not show enough usable separation to
  beat dense under this protocol.
- `soft_moe_E8` is scientifically important but not a performance winner: `20.611%` validation,
  `38.043%` test, and **10.902%** worst-batch accuracy. Its route reliance is `0.0660` (above the
  predeclared `0.01` gate), it uses all eight experts, and expert-output cosine is `0.033` rather
  than approximately one. Routing is therefore genuinely consequential and experts differ; the
  remaining problem is a mean-versus-tail tradeoff, not merely a dead router.
- Conditional-statistic, low-rank, annealed-low-rank, and GroupDRO arms do not improve mean OOD
  validation. The GroupDRO row collapsed (`5.013%` validation) and is an implementation/protocol
  negative, not usable supporting evidence.
- BTX specialists have **no scientific result**: their process failed before training at
  `compute_environment_descriptors` with `KeyError: 'sites'`. Do not include it in any table or
  conclusion until repaired and rerun.

### Current conclusion and next gate

We now have two distinct findings:

1. **Shared residual sparse capacity can modestly improve Cell-DINO adaptation.** The direct
   shared-versus-replacement result is the current best performance signal.
2. **Task-relevant routing can be made real but has not improved mean OOD transfer.** `soft_moe_E8`
   rules out the old explanation that routing was simply inert, yet it loses validation mean while
   improving the worst batch.

The next scientific action should not be another broad architecture grid. First replicate the
frozen `shared_E3k1_late2` recipe against its matched replacement and dense references. If the
mean gain replicates, audit whether it is due to sparse residual capacity or routing; if the
soft-routing tail effect is pursued, make that an explicit mean–tail objective rather than treating
its test score as a win. BTX is a separate engineering repair decision, not evidence.

## Project question now

We are studying **supervised adaptation of a microscopy-pretrained foundation model under
acquisition shift**. Cell-DINO ViT-S/8 is fully fine-tuned from Cell Painting pretraining to RxRx1
genetic-perturbation classification, then evaluated on held-out experimental batches.

The current question is deliberately one level above an RxRx1 leaderboard entry:

> Can sparse conditional residual capacity improve held-out-batch transfer when a
> microscopy-pretrained encoder is supervisedly adapted to a batch-sensitive task?

RxRx1 is the single primary setting. The aim is performance first, with enough controls and
mechanism evidence to explain *why* an MoE helps or fails. Broad cross-domain claims are not
licensed by this study.

## Most recent completed performance table

The following is the latest consolidated terminal table discussed before the shared/residual wave.
It is exploratory single-seed evidence. OOD validation is the selection split; OOD test values are
descriptive readouts for this fixed set of predefined arms, not a basis for post-hoc configuration
search.

| Model | OOD val | OOD test | Worst test batch |
|---|---:|---:|---:|
| Dense expansion, blocks 10–11 | **21.514%** | **38.731%** | 7.951% |
| MoE, block 11 | 20.783% | 36.817% | 6.926% |
| Original Cell-DINO | 20.154% | 36.524% | 6.352% |
| Frozen MoE, block 11 | 20.490% | 36.463% | 6.885% |
| MoE, blocks 10–11 | 20.134% | 36.306% | 8.361% |
| Frozen MoE, blocks 10–11 | 20.317% | 35.818% | 8.279% |
| MoE, blocks 8–11 | 19.576% | 35.810% | 8.689% |
| MoE, all 12 blocks | 19.540% | 35.336% | **9.303%** |

### What the table actually establishes

1. **Dense expansion is the current mean-accuracy baseline to beat.** Relative to original
   Cell-DINO it gains `+1.360` OOD-validation, `+2.207` OOD-test, and `+1.599` worst-batch points.
   The best replacement MoE (block 11) remains `0.731/1.914/1.025` points behind dense on those
   metrics.
2. **Traditional replacement upcycling has not produced the desired OOD gain.** Block-11 MoE is
   only `+0.629` validation and `+0.293` test points above original Cell-DINO. That is too small to
   carry a performance or ICLR-level claim, especially at one seed.
3. **Learned routing is not yet the source of improvement.** Learned versus frozen block-11 MoE is
   only `+0.293/+0.354/+0.041` validation/test/worst points. At blocks 10–11 the comparison is
   `-0.183/+0.488/+0.082`. These small, mixed differences are consistent with expert capacity,
   partitioning, or optimization effects rather than useful adaptive routing.
4. **More replacement depth trades mean accuracy for tail accuracy.** From block 11 to all 12
   blocks, OOD test drops `1.481` points while worst-batch accuracy rises `2.377` points. That is an
   interesting mean–tail tradeoff, not evidence that deeper MoE is better overall.
5. **The bottleneck is transfer, not fitting.** Near-saturated training accuracy in the full
   fine-tuning regime shows that the perturbation labels can be fit. The failure is that the
   solution does not transfer cleanly across experiments/batches.
6. **Validation and test differ greatly in absolute difficulty.** Test being much higher than
   validation does not imply leakage or a conventional learning-curve improvement; the held-out
   experiment sets have different difficulty. Comparisons remain paired within each split.

### What it does not establish

- It does not show that MoE is fundamentally unsuitable for RxRx1.
- It does not show that expert routing is useless under a design that preserves the pretrained
  dense path.
- It does not support selecting architectures from OOD test or claiming a replicated gain.
- Runs made before the routing fix remain preliminary. The new in-wave replacement arm is the
  clean reference for the current implementation.

## Main realization and design change

The prior architecture asked sparse experts to **replace** Cell-DINO FFNs. In this data-constrained
regime, each class has very little within-batch support, so replacement can discard a useful shared
microscopy representation while starving individual experts of stable supervision.

The new architecture keeps the pretrained dense FFN active for every example and adds routed
experts as a residual correction:

`output = pretrained shared FFN(x) + routed sparse correction(x)`

The correction is initialized to zero, so the model starts as exact Cell-DINO rather than a newly
partitioned approximation. This is a standard shared-expert/residual-MoE idea, not a microscopy-
specific bespoke trick. Its intended roles are:

- preserve the common biological representation;
- let sparse capacity model conditional residuals without relearning the whole FFN;
- give every example a high-data shared path in a low-data-per-batch regime;
- test whether batch/morphology heterogeneity is useful after common structure is retained.

This reframes the immediate empirical question from “does generic sparse widening work?” to
“does sparse correction work when destructive replacement is removed?”

## Shared/residual wave design (completed)

Campaign: `shared_residual_performance30_20260807`

All arms use the same Cell-DINO/RxRx1 adaptation protocol, seed 0, 30 epochs, OOD-validation
selection, and terminal readout. Six arms isolate standard MoE design choices; two separately test
established batch-robustness interventions.

| Arm | Primary question |
|---|---|
| `replace_E4k2_late2` | Clean traditional replacement reference under the current code |
| `shared_E3k1_late2` | Does retaining the dense path improve the basic sparse adaptation result? |
| `shared_E3k2_late2` | Does top-2 routing reduce expert starvation or add useful active capacity? |
| `shared_E7k1_late2` | Does more sparse specialization help at fixed top-1 routing? |
| `shared_E3k1_late4` | Does shared/residual MoE tolerate or benefit from greater depth? |
| `shared_E3k2_late4` | Is any depth effect dependent on top-2 routing? |
| `shared_E3k1_xbatch` | Does explicit same-perturbation consistency across experiments improve transfer? |
| `shared_E3k1_mixstyle` | Are feature-statistic/style shifts a major correctable batch bottleneck? |

### Decision map

- **Shared beats replacement and dense:** preserving common pretrained computation was the missing
  design ingredient; proceed to seed replication and routing/mechanism audits.
- **Top-2 beats top-1:** sparse supervision or active capacity was limiting; inspect expert usage
  and determine whether the gain is conditional rather than ordinary extra compute.
- **More experts help only with top-1:** specialization capacity was limiting; verify noncollapse
  and held-out-experiment reuse.
- **Late-4 helps only after adding the shared path:** prior depth failures were caused by destructive
  replacement, not by conditional depth itself.
- **Cross-batch consistency wins:** the dominant problem is alignment of the same perturbation
  across experiments; routing alone is insufficient.
- **MixStyle wins:** batch acquisition statistics are a major transferable nuisance and should be
  integrated with, or compared directly against, conditional capacity.
- **Dense still wins:** current MoE does not add value beyond ordinary capacity. Stop expanding the
  routing grid and redirect effort toward stronger dense adaptation or direct batch-robustness
  objectives.

The aspirational “move-the-needle” region is at least `30%` OOD validation and `40%` OOD test.
These are performance targets, not statistical success thresholds. A paper claim still requires a
material paired gain over dense, no unacceptable ID/tail regression, fresh seeds, and evidence
that routing—not merely added parameters—causes the improvement.

## Historical launch record

Verified on 2026-08-07 after launch:

| Arm | State at first stable global check |
|---|---|
| `replace_E4k2_late2` | training, epoch 2 |
| `shared_E3k1_late2` | training, epoch 2 |
| `shared_E3k2_late2` | training, epoch 1 |
| `shared_E7k1_late2` | training, epoch 1 |
| `shared_E3k1_late4` | training, epoch 2 |
| `shared_E3k2_late4` | training, epoch 2 |
| `shared_E3k1_xbatch` | training, epoch 1 |
| `shared_E3k1_mixstyle` | training, epoch 1 |

- Four SciServer containers each run two H100 jobs; all eight `[start]` events were observed.
- Exact code: branch `agent/shared-residual-performance-wave`, commit `d7fad7a`.
- Clean SciServer worktree:
  `/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation-shared-wave`.
- Results:
  `/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/substrate_rxrx1/cell_dino_cp5/shared_residual_performance30_20260807`.
- Runtime: persistent Python 3.10.20 environment with PyTorch 2.1.0+cu118. The default Python 3.9
  environment is incompatible with vendored DINOv2 annotations.
- W&B is currently offline and Hugging Face upload is deferred because the containers had no
  configured tracking credentials. Persistent JSON/JSONL/checkpoints are the source of truth and
  must be synced later.

Print the global table from a separate SciServer terminal:

```bash
cd /home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation-shared-wave
/home/idies/workspace/Storage/lchen5/persistent/envs/moe/bin/python \
  scripts/sweep_rxrx1_shared_performance.py --status
```

## Next analysis after completion

1. Validate all eight terminal artifacts and protocol fields before ranking anything.
2. Produce one table with OOD validation, OOD test, worst test batch, ID, and train accuracy.
3. Compare the predefined pairs above; do not invent a new grid from test outcomes.
4. Audit expert utilization, entropy, route dependence/randomization, and experiment-conditioned
   usage for any competitive shared-MoE arm.
5. Inspect per-experiment deltas to distinguish broad transfer from rescuing or sacrificing a few
   batches.
6. Replicate only a materially competitive frozen recipe. Treat one-seed differences below a few
   points as design signal, not a result.

## Bottom line

The project has moved from a broad “MoE for batch effects” exploration to a focused performance
and mechanism test in one justified setting. The completed evidence says dense capacity currently
wins mean OOD accuracy and learned replacement routing contributes little. The active wave tests
the most natural remaining MoE hypothesis: preserve the pretrained shared computation and use
sparse experts as conditional corrections. This is a credible, falsifiable next step; it is not
yet a positive result.
