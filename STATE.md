# Current scientific state

Last updated: 2026-08-09 EDT

This is the compact source of truth for the current question, evidence, interpretation, and active
experiment. `PROGRESS.md` remains the chronological ledger; older exploratory analyses remain in
`analysis/`.

## 2026-08-09 larger-data substrate — a real cross-source JUMP task is now specified

An audit of canonical JUMP-CP metadata at commit `016e865` replaces the vague "larger Cell
Painting" idea with a reproducible supervised task. Among non-control compounds, **11,423
compound identities occur in each of sources 1, 2, 3, 8, and 10**. This supports a fixed
cross-source perturbation-identification protocol: train on `source_1/2/3`, select on OOD
validation `source_8`, and reserve `source_10` for descriptive fixed-arm test readout. Gene ORF
and CRISPR identities do not have comparable cross-source replication in this metadata, so they
are not valid substrates for this particular domain-generalization design.

The predeclared scaling ladder is 1,024 / 4,096 / 11,423 classes, initially balanced to one
deterministic well per class per source and one site per well. That becomes 5,120 / 20,480 /
57,115 fields after image sites are resolved. Controls are excluded, class and well selection are
stable-hash deterministic, and the canonical Cell-DINO order is `[DNA, ER, RNA, AGP, Mito]`.
Each ladder will compare original Cell-DINO, matched dense expansion, replacement MoE, and
shared-residual MoE under the same optimizer and seeds; OOD validation remains the selector.

`scripts/build_jump_compound_cross_source_manifest.py` now builds and hashes these metadata-only
manifests, and `docs/jump_cp_scaling_substrate.md` freezes the task, split, controls, channels,
license, acquisition gate, and evaluation. The builder was verified both on a synthetic fixture
and against the current official metadata: it returns exactly 11,423 common non-control compounds
and 57,115 source/well rows for the all-class ladder.

This is not training-ready yet. Neither a JUMP load-data/image index nor raw JUMP pixels are present
in the verified SciServer inventory. The full raw collection is roughly 126.8 TB, and the balanced
one-site all-class subset is only a preliminary ~0.9 TB estimate. The next gate is to join the
manifests to the official image index, audit stain semantics, and compute exact object/byte counts
before requesting storage or downloading pixels. No duplicated or resampled RxRx1 data are being
called a scaling experiment.

The index acquisition is now bounded rather than vague. The official bucket has 1,025 canonical
`load_data.csv` files across the five sources totaling exactly 3,624,396,427 bytes. A sampled
official index exposes source/plate/well/site plus explicitly stain-named five-channel URLs.
`scripts/resolve_jump_image_manifest.py` deterministically selects a complete site and writes URLs
in Cell-DINO order; its fixture test verifies semantic ordering independent of CSV column order.
Only index staging plus an exact selected-object byte audit remain before a pixel-storage decision.

## 2026-08-09 neighbor wave complete — E3 and the late-block pair replicate

`shared_neighbors30_20260809` is 8/8 terminal. OOD validation remains the decision metric; the
completed same-seed E3/top-1 late-2 rows are the predeclared anchors.

| Seed | Shared-residual arm | Train | ID | OOD val | OOD test | Worst test batch | Route reliance |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | E3/top-1, blocks 10–11 (anchor) | 100.000% | **55.612%** | **22.346%** | 38.758% | **8.811%** | 0.0466 |
| 1 | E2/top-1, blocks 10–11 | 100.000% | 55.107% | 21.839% | **38.795%** | 8.115% | 0.0242 |
| 1 | E4/top-1, blocks 10–11 | 100.000% | 55.262% | 21.778% | 38.365% | 8.566% | 0.0467 |
| 1 | E3/top-1, block 10 only | 100.000% | 53.957% | 21.057% | 38.127% | 6.598% | 0.0234 |
| 1 | E3/top-1, block 11 only | 100.000% | 54.115% | 21.798% | 38.101% | 7.623% | 0.0182 |
| 2 | E3/top-1, blocks 10–11 (anchor) | 100.000% | **55.203%** | **22.255%** | 38.833% | 7.910% | 0.0558 |
| 2 | E2/top-1, blocks 10–11 | 100.000% | **55.592%** | 22.032% | 38.813% | 7.336% | 0.0312 |
| 2 | E4/top-1, blocks 10–11 | 100.000% | 54.994% | 22.123% | **38.845%** | **8.197%** | 0.0498 |
| 2 | E3/top-1, block 10 only | 100.000% | 54.134% | 22.082% | 37.965% | 7.910% | 0.0247 |
| 2 | E3/top-1, block 11 only | 100.000% | 54.575% | 21.981% | 38.418% | 7.418% | 0.0137 |

Across seeds, validation mean ± SEM is `22.301±0.046%` for E3/two-block, `21.935±0.096%`
for E2, `21.950±0.173%` for E4, `21.570±0.512%` for block 10, and `21.890±0.091%` for block 11.
E3 beats both E2 and E4 in each seed; its paired mean advantages are `+0.365/+0.351` points at
constant top-1 active compute. The E2-versus-E4 order itself reverses by seed, so neither more nor
less inactive capacity is monotonically beneficial. The two-block E3 anchor also beats both
singletons in each seed, by mean paired margins of `+0.731` over block 10 and `+0.411` over block
11. This supports complementary late-block corrections; a single routed block is insufficient.
Test and tail differences are mixed and descriptive, and none of these neighbors changes the
overall conclusion that the gain is small.

### Predeclared routing-control wave

The completed factor sweep freezes E3/top-1 at blocks 10–11. The next bounded wave is
`shared_routing30_20260809`, exactly four standard routing controls at seeds 1/2, compared with
the completed same-seed anchors. It changes no expert count, placement, optimizer, or active
capacity.

| Arm per seed | Question |
|---|---|
| `shared_E3k1_image` | Does one field-level decision outperform independent token routing? |
| `shared_E3k1_balance1e3` | Does a tenfold weaker load-balance loss permit useful specialization? |
| `shared_E3k1_balance0` | Is the load-balance term helping or constraining the task-trained router? |
| `shared_E3k1_router_frozen` | Does router learning beat an identical fixed random partition during training? |

The first arm tests routing granularity; the middle pair is a two-point regularization ablation;
the last is the direct training-time conditional-routing control missing from the inference-only
random-route audit. OOD validation selects; test remains a fixed descriptive readout. A minimal
`router_frozen` flag for shared-residual MoE freezes only router parameters and leaves expert and
shared-path training unchanged. No new batch regularizer or bespoke expert design is included.

Launch is verified from immutable commit `affb51b` in persistent worktree
`moe-sparse-adaptation-shared-routing-affb51b` after 39 targeted routing/capacity tests passed.
Container `2859` runs shard 0 and `2862` shard 1; each printed two distinct starts on GPUs 0/1.
All four first-wave processes hold 8.5–8.7 GB, logs show RxRx1/Cell-DINO initialization without
tracebacks, the image and zero-balance arms have written epoch 1, and the other two are completing
first-epoch initialization. Seed 2 is queued behind the same healthy controllers. Tracking is
local-first because W&B/HF credentials remain absent. Results persist at
`substrate_rxrx1/cell_dino_cp5/shared_routing30_20260809`.

## 2026-08-09 mechanism result — routing is causal, but the gain is not hardest-batch rescue

The two checkpoint-only mechanism audits and both class-matched geometry reports completed
cleanly under `shared_confirm30_20260809/diagnostics_hb8`.

| Seed | Full OOD val | Random-route val | Route reliance | Correction-off val | Residual contribution |
|---:|---:|---:|---:|---:|---:|
| 1 | 22.346% | 17.688% | **4.658 pt** | 19.180% | **3.166 pt** |
| 2 | 22.255% | 16.673% | **5.581 pt** | 18.662% | **3.592 pt** |

This decisively rejects an inert-router or ordinary-capacity explanation for the trained shared
checkpoint. The sparse correction is necessary, and learned conditional assignment is necessary:
randomly reassigning the already-trained experts costs about five validation points in both seeds.
Correction-off performance is also below the separately trained original Cell-DINO arm, so the
full network has co-adapted with the residual path; `full - correction-off` should not be read as
an independently additive improvement over the original model.

All three experts are used at both routed blocks and routing entropy is near maximal, ruling out
expert collapse. Block-10 routing is more acquisition-site aligned than class aligned in both
seeds (site MI `0.103/0.246` versus class MI `0.030/0.024`); block 11 is weaker and more mixed
(site MI `0.036/0.126`, class MI `0.062/0.046`). This makes placement a concrete next question:
block 10 may supply useful batch-conditional correction, or it may encode a fragile batch
shortcut.

The representation diagnostics narrow the claim. Dense and shared make highly overlapping errors
(validation error Jaccard `0.908/0.924`; test `0.824/0.814`), so shared is moving a modest decision
boundary rather than solving a new failure mode. Shared's mean validation retrieval advantage over
dense is only `+0.59` points and its test retrieval is `-0.31`; global CKA to frozen Cell-DINO is
slightly lower than dense in both seeds. Batch-variance fraction is slightly lower for shared, but
class variance also falls, so this is not clean batch removal. Accuracy remains strongly negatively
associated with independently defined frozen-space OOD severity, and shared's correlation is not
flatter than dense (`-0.670/-0.624` versus `-0.634/-0.580`). The hardest validation experiments
remain near floor. The supported interpretation is **efficient conditional adaptation with a real
router**, not general embedding debatching or disproportionate rescue of the most OOD batches.

### Predeclared bounded neighbor wave

The next wave is `shared_neighbors30_20260809`: exactly eight 30-epoch runs, four standard shared-
residual neighbors at the same two seeds as the frozen E3/top-1 late2 anchor. Active expert compute
stays top-1; OOD validation decides and all test reads are fixed before launch.

| Arm per seed | Question |
|---|---|
| `shared_E2k1_late2` | Does less inactive expert capacity outperform E3 at constant active compute? |
| `shared_E4k1_late2` | Does one more expert improve specialization at constant active compute? |
| `shared_E3k1_block10` | Is the more batch-aligned block-10 router sufficient or shortcut-prone? |
| `shared_E3k1_block11` | Can the later, less site-aligned route retain the gain with less disruption? |

This is two bounded variations of expert-bank size and two of placement, not a new broad grid. It
reuses seeds 1/2 so every arm has a same-seed comparison to completed `shared_E3k1_late2`. A
one-command status/aggregate report includes those frozen anchors. No new batch regularizer is
included because MixStyle and cross-experiment consistency already hurt, while the geometry audit
does not support a simple debatching objective.

Launch is verified from immutable commit `4f032e0` after all eight targeted wave/aggregator tests
passed on SciServer. Container `2862` runs shard 0 and `2859` runs shard 1; each printed two
distinct starts on GPUs 0/1 with no immediate failure. The four seed-1 rows reached epoch 1 and
the global table advanced to training; the four seed-2 rows are queued inside the two healthy
foreground sweep controllers. Tracking is local-first because W&B/HF credentials remain absent.
Results persist at `substrate_rxrx1/cell_dino_cp5/shared_neighbors30_20260809`.

## 2026-08-09 steward result — shared residual wins both fresh-seed validation comparisons

Campaign `shared_confirm30_20260809` is complete: eight terminal 30-epoch rows, two fresh seeds,
and the four predeclared matched arms. OOD validation is the decision metric; OOD test is a
descriptive readout of these fixed checkpoints.

| Seed | Arm | Train | ID | OOD val | OOD test | Worst test batch | Route reliance |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Original Cell-DINO | 100.000% | 52.581% | 20.459% | 36.454% | 6.639% | — |
| 1 | Dense E4, late 2 | 100.000% | 55.188% | 21.839% | **38.981%** | 7.869% | — |
| 1 | Replacement E4/top-2, late 2 | 100.000% | 54.223% | 20.926% | 37.715% | 6.844% | 0.0507 |
| 1 | Shared residual E3/top-1, late 2 | 100.000% | **55.612%** | **22.346%** | 38.758% | **8.811%** | 0.0466 |
| 2 | Original Cell-DINO | 100.000% | 52.731% | 20.875% | 37.131% | 7.213% | — |
| 2 | Dense E4, late 2 | 100.000% | 54.742% | 21.737% | **38.877%** | **7.910%** | — |
| 2 | Replacement E4/top-2, late 2 | 100.000% | 53.753% | 21.240% | 38.011% | 7.705% | 0.0439 |
| 2 | Shared residual E3/top-1, late 2 | 100.000% | **55.203%** | **22.255%** | 38.833% | **7.910%** | 0.0558 |

Across the two fresh seeds, OOD-validation mean ± SEM is `20.667±0.208` for original,
`21.788±0.051` for dense, `21.083±0.157` for replacement, and `22.301±0.046` for shared
residual. Shared beats dense by `+0.507/+0.518` validation points and replacement by
`+1.421/+1.015` in seeds 1/2. The paired mean gains are therefore `+0.513` versus dense and
`+1.218` versus replacement. Its mean paired differences versus dense are `-0.134` on descriptive
test, `+0.472` on worst test batch, and `+0.442` on ID; versus replacement they are
`+0.933/+1.086/+1.420`. The earlier seed-0 campaign had the same validation direction
(`+0.728` versus dense and `+0.474` versus replacement), although it was a different campaign and
is supporting rather than pooled evidence.

This is promising and reproducible as a small effect, not yet a large performance breakthrough.
It rejects the claim that ordinary dense widening is the whole story: shared residual wins both
fresh-seed validation comparisons while activating `4,728,578` FFN parameters versus dense's
`9,454,854`, at matched total capacity near `29.494M`. It also cleanly beats equally active,
equally capacious replacement MoE, supporting preservation of Cell-DINO's pretrained dense path.
The descriptive test mean remains essentially tied with dense and no arm reaches `30%` validation
or `40%` test, so the present claim is efficient held-out-batch adaptation rather than SOTA.

With the two scoped containers verified free, four checkpoint-only diagnostics were launched from
immutable commit `9365406`: shared correction-off/random-route mechanism audits for seeds 1 and 2,
plus class-matched batch/embedding geometry reports comparing all four arms separately at each
seed. All four processes are healthy on the four bottom-container H100s with no immediate error.
They write to `shared_confirm30_20260809/diagnostics_hb8`. These tests ask whether the routed
correction itself and learned assignments matter, and whether shared residual changes degradation
with independently defined OOD severity, batch-versus-class geometry, cross-batch retrieval,
confidence, or error overlap. No new architecture wave is justified until these diagnostics
separate mechanism from capacity and representation preservation.

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

At this interim snapshot both sweep controllers were healthy and all four seed-2 arms were
training. Their completed results are recorded in the section above.

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
