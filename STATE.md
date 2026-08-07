# Current scientific state

Last updated: 2026-08-07 EDT

This is the compact source of truth for the current question, evidence, interpretation, and active
experiment. `PROGRESS.md` remains the chronological ledger; older exploratory analyses remain in
`analysis/`.

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

## Active eight-arm performance wave

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

## Live operational state

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
