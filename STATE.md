# Current scientific state

Last updated: 2026-08-10 EDT

This is the compact source of truth for the current question, evidence, interpretation, and active
experiment. `PROGRESS.md` remains the chronological ledger; older exploratory analyses remain in
`analysis/`.

## 2026-08-10 RxRx3 plate-count curve: one-plate endpoint predeclared

Launch gate is now closed and the endpoint is active. Immutable execution source
`3f9aca5889b08d173775ea77ee460d944be28db4` lives at
`/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation-rxrx3-plate1-3f9aca5`.
In SciServer Python 3.10 it passed entrypoint compilation, 17 focused RxRx3 loader/manifest/audit/
sweeper tests, and all 344 regression tests. The real-data dry run reverified eight unique run IDs,
two jobs on each of four shards, a clean exact-source manifest, 2,696/2,708/23,855 train/ID/OOD
wells, strict 1/2/4/8 train-well nesting, fixed evaluation wells, the six-channel pixel gate, and
zero pre-existing result artifacts.

All four distinct 2xH100 containers were then proven free of GPU applications and experiment
processes. The eight predeclared jobs launched two per container on GPUs 0/1 in local-first
tracking mode. Eight distinct start lines and child PIDs were observed, the global status command
shows all eight arms training at epoch 1 or 2, and a recursive launch-log scan found no traceback,
OOM, kill, segmentation fault, or `rc=1` signature. Results and the one-command manifest/status
view are under
`substrate_rxrx3_core/cell_dino_cp5/rxrx3_core_plate1_scale10_20260810`.

The competence pilot passed, so the first honest RxRx3 scaling axis is now frozen. The scientific
question is whether the OOD-test architecture gap changes as **train plates per experiment** grow
from 1 to 2 to 4 to 8 while four guides per gene, 674 labels, 85 train experiments, 85 paired
held-out experiments, ID-validation wells, OOD-test wells, preprocessing, optimization, and
checkpointing remain fixed. Plate count and examples per class necessarily co-vary on this axis;
it is not called pure sample scaling or experiment-count scaling. The atomic sampling unit remains
one six-channel well, and the four manifests must form strictly nested train-well sets with
identical evaluation-well sets.

The first new wave is the one-plate endpoint: exactly eight jobs, comprising the unchanged
original, dense E4 late-2, replacement E4/top-2 late-2, and shared-residual E3/top-1 late-2 arms at
seeds 1/2. The completed eight-plate competence pilot is the protocol-matched full-data anchor;
two- and four-plate points will be separate later waves. OOD-test accuracy and worst held-out
experiment accuracy are headline. The primary estimand is the shared-minus-dense change over
`log2(plate count)`; shared-minus-replacement and dense-minus-original are secondary mechanism
contrasts. Material claims require the full 1/2/4/8 curve, paired per-experiment differences,
seed replication, and hierarchical uncertainty. This endpoint alone cannot establish a scaling
interaction.

All jobs stop at epoch 10 and use fixed ID-validation plates plus the terminal checkpoint; test
does not select topology. Only `seed`, `run_tag`, `rxrx3_manifest`, and the declared architecture
fields may differ. Total/active FFN parameters remain 22,044,578/2,363,136 original,
29,136,296/9,454,854 dense, 29,137,060/4,729,346 replacement, and
29,136,292/4,728,578 shared. Thus dense and shared are matched in total capacity to four
parameters, while shared activates about half the dense FFN parameters.

The predeclared result directory is
`substrate_rxrx3_core/cell_dino_cp5/rxrx3_core_plate1_scale10_20260810`. The new launcher audits
all four plate manifests together, requires 2,696/5,376/10,706/21,404 nested train wells, freezes
2,708 ID and 23,855 OOD wells, rejects split or coverage drift, and provides the same one-command
status table as the pilot. Before launch, its exact clean source must pass SciServer compilation,
focused tests, all regression tests, real-manifest audit, resolved-config comparison, dry-run
sharding, and eight-GPU freedom checks.

## 2026-08-10 RxRx3-core loader passes its first gate; competence pilot predeclared

The training path is now explicit rather than inferred from the metadata audit. A Parquet-backed
dataset indexes only the selected channel keys, lazily reads the row groups required by each well,
decodes exactly six grayscale acquisitions, applies one joint geometric transform, maps
`[w1,w2,w4,mean(w3,w6),w5]` before per-channel standardization, and returns both the contiguous
train-site ID and a stable raw experiment ID. A manifest-scoped persistent index prevents eight
jobs from rescanning 1.336M keys, while a row-group-local shuffle preserves stochastic training
without rereading a roughly 1.3-MB/100-channel row group for every well. Missing, duplicate, or
corrupt channels; label/split drift; incomplete class coverage; and experiment leakage fail
closed. RxRx1 dispatch and OOD-validation semantics remain unchanged; RxRx3 explicitly records
its frozen checkpoint split as `id_val`.

Final execution source `79f19f5` passed compilation, 27 focused loader/runner/native-channel/
pilot-sweeper tests, and all 341 tests in the SciServer Python 3.10 environment. Its exact-source
real-data smoke resolved the frozen
21,404/2,708/23,855 train/ID-validation/OOD-test rows, emitted `Bx5x128x128` finite tensors with
train sites non-negative and every OOD site equal to `-1`, and decoded four batches from every
split. Eight-worker throughput was about 49/65/64 wells/s for train/ID/OOD. The immutable
worktree is `moe-sparse-adaptation-rxrx3-pilot-79f19f5`; the exact-source dry run passed the
manifest, data, resolved-config, capacity, unique-ID, and sharding gates.

The first GPU wave is now predeclared as a **competence pilot**, not a scaling result. The atomic
unit is one six-channel well. Data are the full eight-train-plate/four-guide manifest with SHA-256
`131815361655c6f795929bee6de5da5f249bacad8ded50f20e3a036243d5af2f`: 674 fixed classes,
85 train experiments, 85 disjoint paired OOD-test experiments, one fixed ID-validation plate per
train experiment, and identical evaluation rows in every arm. Exactly eight jobs use seeds 1/2
and the established matched architectures:

| Arm | Architecture | Scientific role |
|---|---|---|
| `original` | Original Cell-DINO-S; blocks 10--11 identified but unchanged | Pretrained supervised-adaptation baseline |
| `dense_E4_late2` | Dense E4 widening in blocks 10--11 | Ordinary added-capacity control |
| `replace_E4k2_late2` | Replacement sparse E4/top-2 in blocks 10--11 | Traditional sparse-upcycling control |
| `shared_E3k1_late2` | Shared-residual E3/top-1 in blocks 10--11 | Pretrained-path-preserving sparse model |

Every non-architecture field is programmatically matched: channel map, splits, labels,
augmentation, batch size 64, optimizer, learning rate, weight decay, warmup, LLRD, drop path,
losses, 10-epoch terminal checkpoint/readout, and stage-3 evaluation. The primary contrast is
shared residual versus dense at matched total capacity; shared versus replacement isolates path
preservation and dense versus original isolates generic capacity. OOD-test accuracy,
worst-OOD-experiment accuracy, and total/active parameter and relative active-FFN-FLOP accounting
are headline; the fixed ID plate is only the internal checkpoint/competence readout. The pilot
stops after exactly 10 epochs for all eight arms, with no adaptive topology addition. The
predeclared competence gate is finite artifacts with train accuracy at least 5% and ID validation
at least 1% for every arm. Only if it passes do the separate 1/2/4/8-plate and 1/2/4-guide curves
become launchable.

Execution completed from the immutable source on 2026-08-10. All eight jobs are terminal with
valid stage-3 artifacts and exact manifest/source/config identity. The full per-seed table is:

| Seed | Arm | Train | ID validation | OOD test | Worst OOD experiment |
|---:|---|---:|---:|---:|---:|
| 1 | Original | 27.423% | 16.322% | 1.966% | 0.000% |
| 1 | Dense E4 late-2 | 28.588% | 16.581% | 1.966% | 0.000% |
| 1 | Replacement E4/top-2 late-2 | 27.456% | 15.879% | 1.748% | 0.000% |
| 1 | Shared E3/top-1 late-2 | 26.670% | 16.211% | 1.953% | 0.000% |
| 2 | Original | 28.069% | 15.733% | 2.083% | 0.000% |
| 2 | Dense E4 late-2 | 30.838% | 17.097% | 2.130% | 0.000% |
| 2 | Replacement E4/top-2 late-2 | 28.698% | 15.953% | 1.874% | 0.000% |
| 2 | Shared E3/top-1 late-2 | 27.358% | 15.177% | 1.882% | 0.000% |

The corresponding descriptive two-seed means are:

| Arm | Train | ID validation | OOD test | Worst OOD experiment |
|---|---:|---:|---:|---:|
| Original | 27.746% | 16.027% | 2.025% | 0.000% |
| Dense E4 late-2 | 29.713% | 16.839% | 2.048% | 0.000% |
| Replacement E4/top-2 late-2 | 28.077% | 15.916% | 1.811% | 0.000% |
| Shared E3/top-1 late-2 | 27.014% | 15.694% | 1.918% | 0.000% |

Thus the task is learnable but sharply out-of-experiment: every arm clears the frozen train/ID
competence thresholds, yet mean OOD test is only about 2% and every architecture has at least one
zero-accuracy held-out experiment. At this budget, shared trails dense by 0.130 OOD-test points
and 1.145 ID points, while exceeding replacement by 0.107 OOD-test points. Dense exceeds original
by only 0.023 OOD-test points. These small two-seed pilot gaps are descriptive, not a material
architecture claim; the pilot's decision is simply that the loader, task, optimization, and
stage-3 evaluation are competent enough to license the predeclared within-RxRx3 scaling curves.

Once those six controllers exited, three containers were re-audited at zero experiment processes
and zero GPU applications. Replacement E4/top-2 seeds 1/2 were then launched as the predeclared
shard 2 on one proven-idle 2xH100 container. Both reached epoch 3, after which their Jupyter
endpoint returned HTTP 503 and both train logs stopped changing for more than an hour. This is an
infrastructure interruption rather than a scientific artifact. The six partial files were moved
intact to `recovery_dead_container_20260810T1515EDT` under the persistent result directory so that
duplicate epoch records cannot contaminate the rerun. A second container was proved free by both
an empty GPU-application query and an empty experiment-process query, then the same exact-source
shard was relaunched without changing its manifest rows, run IDs, configs, or protocol. Both
recovery rows then completed epoch 10 plus ID and OOD evaluation, with two distinct starts, live
GPU/process verification, clean controller exits, and no fatal signature. Results live at
`substrate_rxrx3_core/cell_dino_cp5/rxrx3_core_pilot10_20260810`, with local-first tracking.

## 2026-08-10 RxRx3-core is viable, but its honest scaling axes differ from RxRx1

The official RxRx3-core release is now located and its complete 24,860,992-byte metadata table has
been audited (SHA-256 `0f69ea7d2122c0c2ccace20a34fc70f0fec11a06cc4fca5eed550c0bc6985913`).
The release contains 222,601 unique HUVEC wells, 1,335,606 channel-image rows, 1,744
experiment-plate pairs, 736 CRISPR targets, and 1,674 compounds. The 35 image Parquet shards total
about 17.4 GB; the 23.1-GB repository total also includes unrelated pretrained embeddings and
CellProfiler features. The images are six 512×512 uint8 JPEG-2000 center crops per well, so they
fit the existing fixed native map `[w1,w2,w4,mean(w3,w6),w5]` into Cell-DINO's
`[DNA,ER,RNA,AGP,Mito]` slots.

The key split structure is stronger than the previous inventory suggested. Query-guide metadata
has 734 genes: 695 occur in two experiments, 24 in four, and 15 in one. Of 176 CRISPR experiments,
174 form 87 exact pairs with identical query-gene sets. This licenses a paired supervised task:
train on one member of every pair and evaluate on the independently repeated mate. A deterministic
metadata-only builder now requires four guide IDs shared across paired experiments, reserves one
class-complete train plate for ID validation, and requires a second complete plate so the smallest
train point keeps all classes. Two pairs fail this QC gate. The frozen substrate therefore has:

| Quantity | Audited value |
|---|---:|
| Fixed gene classes | 674 |
| Train / OOD-test experiments | 85 / 85 |
| Train / OOD experiment overlap | 0 |
| Train wells at 1 / 2 / 4 / 8 plates | 2,696 / 5,376 / 10,706 / 21,404 |
| Fixed ID-validation / OOD-test wells | 2,708 / 23,855 |
| Class coverage at every split and scale | 674 / 674 |

This dataset cannot support an honest fixed-label **experiment-count** curve: genes are partitioned
across experiment pairs, so removing experiment pairs also removes classes. Instead, RxRx3-core
supports two distinct nested curves: (1) 1/2/4/8 train plates with four guides fixed, where
plate-batch count and sample density co-vary; and (2) 1/2/4 guides with all eight train plates
fixed, which changes guide diversity/examples per class while holding plate domains fixed. These
axes directly test whether sparse capacity benefits from more independently repeated plates or
from more biological guides, without calling a class-count confound domain scaling.

`scripts/build_rxrx3_core_gene_manifests.py`, `scripts/audit_rxrx3_core_images.py`, and
`docs/rxrx3_core_scaling_substrate.md` freeze the split, curves, channel map, hashes, and pixel
gate. Generated manifests remain outside git because they are licensed data derivatives. The
image-only release is now staged at
`/home/idies/workspace/Storage/lchen5/persistent/datasets/rxrx3-core`: all 35 shards are present at
the exact expected 17,390,577,507 bytes. The exhaustive audit passes all checks over 1,335,606
keys and 222,601 wells: every well has one each of channels 1--6, all 47,967 wells in the union of
the seven manifests resolve, and one decoded image from every shard is 512×512 grayscale with no
decode error. Source `e3a6f61` compiles and passes 14 focused plus native-channel regression tests
in the SciServer Python 3.10 environment.

The data/acquisition gate is therefore closed. The remaining launch blocker is implementation and
testing of the Parquet-backed six-channel dataset loader, exact resolved-config matching, and the
four-arm predeclared pilot—not missing pixels. Recursion's current EULA permits this research use
subject to attribution and restrictions, but its derivative/share-alike terms mean checkpoints
cannot be published under an incompatible license. All four SciServer containers remain at zero
experiment processes and zero GPU compute applications; no GPU job was launched before the loader
and experiment correctness gates.

## 2026-08-10 three-point RxRx1 conclusion: scale strengthens capacity, not shared-over-dense

The predeclared 16-experiment midpoint wave is 8/8 terminal and artifact-valid. All rows are
stage-3 epoch-30 evaluations from clean source `ab309d3`; run identity, seed, checkpoint split,
environment subset, train-accuracy recording, per-batch denominators, worst-batch calculation,
and total/active parameter counts pass. The only literal manifest/result config difference is the
deterministic runtime dataset summary `sites={K:33,n_cell_types:4}`, which is explicitly ignored
by the artifact comparison because it is derived metadata rather than a training factor.

| Seed | Arm | Train | raw ID* | OOD val | OOD test | Worst test batch |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Original Cell-DINO | 100.000% | 25.985% | 9.915% | 18.137% | 3.893% |
| 1 | Dense E4, blocks 10--11 | 100.000% | 28.021% | 10.229% | 19.328% | 4.344% |
| 1 | Replacement E4/top-2 | 100.000% | 27.977% | 10.037% | 19.305% | 4.918% |
| 1 | Shared-residual E3/top-1 | 100.000% | 28.206% | 10.331% | 19.685% | 4.098% |
| 2 | Original Cell-DINO | 100.000% | 25.185% | 10.037% | 17.545% | 3.320% |
| 2 | Dense E4, blocks 10--11 | 100.000% | 27.888% | 10.686% | 19.633% | 3.934% |
| 2 | Replacement E4/top-2 | 100.000% | 27.416% | 10.533% | 19.064% | 4.344% |
| 2 | Shared-residual E3/top-1 | 100.000% | 28.029% | 10.838% | 19.244% | 4.508% |

The three-point paired analysis uses the fixed 8/16/33-experiment curve, seeds 1/2, and 20,000
hierarchical bootstrap draws over paired seeds and the seven held-out test batches. Slopes are
OOD-test architecture-gap percentage points per doubling of training experiments:

| Contrast | Gap @8 | Gap @16 | Gap @33 | Mean slope | Seed+batch 95% CI | Positive test batches |
|---|---:|---:|---:|---:|---:|---:|
| Shared residual − dense widening | +0.494 | −0.016 | −0.134 | **−0.305** | **[−0.598, −0.070]** | 14.3% |
| Shared residual − replacement MoE | +0.212 | +0.280 | +0.932 | **+0.354** | **[+0.043, +0.628]** | 85.7% |
| Dense widening − original Cell-DINO | +0.135 | +1.639 | +2.136 | **+0.975** | **[+0.693, +1.263]** | 100.0% |

Both shared-minus-dense seed slopes are negative (`−0.233/−0.378`). The new midpoint therefore
does not reveal a hidden crossover: shared is useful in the smallest regime, ties dense at the
middle point, and trails slightly at full scale. Conversely, the benefit of preserving the
pretrained dense path over replacement sparse upcycling grows with scale, and ordinary added
capacity grows even more strongly. Midpoint deviations from endpoint log-linear interpolation
are `−0.203/−0.284/+0.526` points for the three contrasts, so the curve is not perfectly linear;
the slope signs, however, are paired and consistent. With only two midpoint seeds this remains a
strong diagnostic rather than a universal scaling-law estimate.

The RxRx1 endpoint is now scientifically clear: shared-residual MoE gives near-dense mean test
accuracy at about half the active FFN parameters and beats replacement upcycling, but increasing
RxRx1 training experiments does not make it outperform dense widening. Another nearby topology
grid or a fixed-update rescue experiment is lower leverage. The next paper-critical step is a
genuinely larger, independently batched substrate (RxRx3-core first, then JUMP-CP), with separate
within-dataset curves and the same four matched arms. A fresh SciServer persistent-storage
inventory found only JUMP code/docs and no RxRx3/JUMP object larger than 1 GB at the searched
depth, so no honest GPU scaling launch was possible in this heartbeat; acquisition/index/channel
and leakage audits remain the gate.

Analysis/audit source `290d281` passed compilation, 10 focused tests, and all 326 tests in the
actual SciServer Python 3.10 environment. Reports live beside the results as
`midpoint_terminal_audit.json`, `three_point_table.md`, and `three_point_uncertainty.{json,md}`
under `substrate_rxrx1/cell_dino_cp5/rxrx1_domain_midpoint30_20260810`. All four containers now
report zero run processes, zero midpoint controllers, and zero GPU compute applications.

*The raw ID column still spans the original training environments and is not scale-comparable.*

## 2026-08-10 three-seed domain-scaling conclusion: preservation scales; shared does not beat dense

The seed-1/2 quarter-scale replication is 8/8 terminal and valid, and its exact-protocol full
anchors are complete. A historical config-audit false positive was repaired: the old artifacts
predate `model.router_frozen`, whose missing value is semantically the same as the current default
`false`. The audit still rejects `true` and every other unexpected drift. Commit `2a6a06e` passes
13 focused and all 316 SciServer Python 3.10 tests; the final aggregate reports all configs and
artifacts passing, eight clean launcher exits, and no fatal signature.

Paired seed and test-batch uncertainty over seeds 1, 2, and 5 changes the interpretation. Values
below are OOD-test percentage points for `(full architecture gap) - (quarter architecture gap)`:

| Contrast | Quarter gap | Full gap | Interaction | Seed 95% CI | Seed+batch 95% CI | Positive test batches |
|---|---:|---:|---:|---:|---:|---:|
| Shared residual − dense widening | +0.402 | +0.042 | **−0.360** | [−0.784, +0.174] | [−0.977, +0.244] | 28.6% |
| Shared residual − replacement MoE | +0.215 | +1.005 | **+0.790** | [+0.444, +0.996] | [+0.315, +1.212] | 100.0% |
| Dense widening − original Cell-DINO | +0.167 | +2.085 | **+1.919** | [+1.754, +2.059] | [+1.384, +2.449] | 100.0% |

The completed evidence supports two effects and rejects one hoped-for story. More training scale
reliably increases the value of added capacity, and retaining the pretrained dense FFN path lets
shared-residual MoE scale substantially better than replacement sparse upcycling. However, shared
does **not** scale better than matched dense widening on mean OOD test: its small advantage is
larger in the data-poor endpoint and disappears at full scale. The defensible result is therefore
near-dense performance at roughly half the active FFN parameters plus better preservation than
traditional replacement upcycling—not a performance win over dense. Analysis commit `0724202`
passes 9 focused and all 319 SciServer tests and writes 20,000-draw paired seed/batch bootstrap
reports into `rxrx1_domain_scaling_replicate30_20260810`.

## 2026-08-10 predeclared 16-experiment midpoint wave

RxRx1's WILDS training metadata has only site value `1` for all 40,612 fields, so it cannot support
a genuine within-experiment site/replicate-density curve. A one-site subset would equal the full
training set and would not change examples per expert. The next honest in-dataset step is therefore
a third point on the already-defined **training-experiment-count** curve, not a mislabeled density
experiment.

Campaign `rxrx1_domain_midpoint30_20260810` contains exactly eight new runs: original Cell-DINO,
dense E4 late-2, replacement E4/top-2 late-2, and shared-residual E3/top-1 late-2 at seeds 1 and 2.
The midpoint is the deterministic, outcome-independent 16-environment nested prefix
`[5,2,4,14,12,17,15,13,21,20,22,41,35,37,46,47]`. It strictly contains the frozen eight-
environment point and is strictly contained in the 33-environment endpoint. The SciServer metadata
audit finds 19,712 fields/wells/sites, all 1,139 classes, at least 15 examples per class, and all
four cell types; per-class counts are 15/16/68 (minimum/median/maximum). OOD evaluation domains,
label set, six-to-five channel adapter, optimizer,
30-epoch checkpoint rule, architecture controls, and seeds are fixed.

| Seed | Original | Dense E4 late-2 | Replacement E4/top-2 | Shared E3/top-1 |
|---:|---|---|---|---|
| 1 | `midpoint_original_s1` | `midpoint_dense_E4_late2_s1` | `midpoint_replace_E4k2_late2_s1` | `midpoint_shared_E3k1_late2_s1` |
| 2 | `midpoint_original_s2` | `midpoint_dense_E4_late2_s2` | `midpoint_replace_E4k2_late2_s2` | `midpoint_shared_E3k1_late2_s2` |

The headline estimand is the per-seed slope of the shared-minus-dense OOD-test gap over
`log2(8,16,33)` training experiments; worst-test-batch is co-headline. Shared-minus-replacement
and dense-minus-original locate preservation and generic-capacity effects. The wave stops after
all eight terminal epoch-30 evaluations and no topology is added adaptively. It improves curve
shape and interaction estimation, but experiment count, fields, and optimizer updates still
co-vary; it does not by itself identify which of those three is causal. The one-command status
view joins the new rows to the completed quarter/full anchors and fails on resolved-config or
artifact drift.

The wave is live from pushed source `ab309d3a14b9859033ad166aba8bb6990899169e` and clean immutable
SciServer worktree
`/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation-domain-midpoint-ab309d3`.
The actual SciServer Python 3.10 runtime passed compilation, 20 focused tests, and all 324 tests.
All four dry-run shards contain exactly two unique rows; the manifest records the exact source SHA,
clean state, resolved configs, total/active parameter accounting, relative active FFN FLOPs, nested
subset hashes/statistics, and zero split-index overlap. Immediately before launch, every container
reported zero controllers, zero training processes, and zero GPU compute applications. Four
controllers then emitted eight distinct starts, two per container and one on each GPU. All eight
run logs have reached real epoch records, all H100s have active workers, and the global fatal scan
is empty. Tracking is local-first because W&B/Hugging Face credentials remain absent. Results and
the one-command table persist under
`substrate_rxrx1/cell_dino_cp5/rxrx1_domain_midpoint30_20260810`.

## 2026-08-10 seed-5 domain-scaling wave complete: weak positive test interaction, mixed endpoints

All eight predeclared artifacts are terminal and valid. The global audit found eight manifest
rows, eight results, eight run logs, eight starts, eight clean exits, and no traceback, OOM,
runtime error, or nonzero exit. All four sweep controllers exited and all eight H100s are free.

| Scale | Arm | Train | Standard `id_test`* | OOD val | OOD test | Worst test batch |
|---|---|---:|---:|---:|---:|---:|
| 8 experiments / 9,856 fields | Original Cell-DINO | 100.000% | 9.049% | 4.343% | 6.044% | 2.664% |
| 8 experiments / 9,856 fields | Dense E4, blocks 10--11 | 100.000% | 9.689% | 4.496% | 6.273% | 2.828% |
| 8 experiments / 9,856 fields | Replacement E4/top-2 | 100.000% | 9.655% | 4.506% | 6.270% | 3.320% |
| 8 experiments / 9,856 fields | Shared-residual E3/top-1 | 100.000% | 10.076% | 4.800% | 6.491% | 2.910% |
| 33 experiments / 40,612 fields | Original Cell-DINO | 100.000% | 52.384% | 20.662% | 36.539% | 6.639% |
| 33 experiments / 40,612 fields | Dense E4, blocks 10--11 | 100.000% | 54.553% | 21.555% | 38.522% | 9.180% |
| 33 experiments / 40,612 fields | Replacement E4/top-2 | 100.000% | 53.876% | 20.936% | 37.764% | 8.402% |
| 33 experiments / 40,612 fields | Shared-residual E3/top-1 | 100.000% | 55.370% | 21.808% | 38.914% | 9.139% |

The shared-minus-dense OOD-test advantage rises from `+0.218` points at quarter scale to
`+0.392` at full scale, an architecture × scale interaction of `+0.174` points. The other
predeclared interactions are mixed: worst-test-batch is `-0.123` and OOD validation is `-0.051`.
This is therefore a weak one-seed positive test signal, not evidence that MoE scales better.
Across the five existing full-data seeds, shared loses to dense in four and the mean paired
test difference is still about `-0.100` points. The scientifically live question is whether the
quarter endpoint and interaction replicate across seeds, not which neighboring topology happens
to win seed 5.

`id_test` is only a raw audit readout because it includes 25 training experiments unseen by the
quarter-scale arms. It must not be used as a scale interaction. Fixed 30 epochs also gives full
scale 4.12× more optimizer updates; a fixed-update control is warranted only if the replicated
architecture × scale interaction is positive.

## 2026-08-10 predeclared seed-1/2 domain-scaling replication

The next bounded wave is `rxrx1_domain_scaling_replicate30_20260810`: exactly eight new jobs,
all at the frozen quarter-scale subset `[5,2,14,12,17,15,41,46]`, with original Cell-DINO,
dense E4 late-2, replacement E4/top-2 late-2, and shared-residual E3/top-1 late-2 at seeds 1 and
2. Their already-completed, exact-protocol full-data counterparts in `shared_confirm30_20260809`
are frozen anchors; no full run is repeated or selected from seed-5 test noise.

| Seed | Original | Dense E4 late-2 | Replacement E4/top-2 | Shared E3/top-1 |
|---:|---|---|---|---|
| 1 | `quarter_original_s1` | `quarter_dense_E4_late2_s1` | `quarter_replace_E4k2_late2_s1` | `quarter_shared_E3k1_late2_s1` |
| 2 | `quarter_original_s2` | `quarter_dense_E4_late2_s2` | `quarter_replace_E4k2_late2_s2` | `quarter_shared_E3k1_late2_s2` |

The primary estimand is the across-seed mean of
`(shared − dense at full) − (shared − dense at quarter)` for OOD test accuracy; worst-test-batch
is the co-headline robustness endpoint. Terminal epoch 30 is fixed, every architecture and split
factor is matched, and all eight rows receive stage-3 evaluation. Total shared/dense capacity is
approximately matched, while shared activates `4.729M` FFN parameters versus dense's `9.455M`.
The stopping rule is the completion of these eight jobs; no topology is added adaptively.

The replication launcher fails on any quarter/full resolved-config drift, audits split-index
disjointness and quarter-scale class/cell-type coverage, writes the full manifest before launch,
and joins new rows to frozen anchors in one status table. Locally, 22 focused tests and the complete
316-test suite pass.

Launch is verified from pushed commit `60ca493` and immutable SciServer worktree
`/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation-domain-replicate-60ca493`.
The actual SciServer Python 3.10 runtime passed the same 22 focused tests and all 316 tests, all
four dry-run shards contained exactly two unique rows, the manifest audit passed, and all eight
GPUs were empty before allocation. Four controllers then emitted eight distinct starts: four on
GPU 0 and four on GPU 1. The global fatal scan is empty, all eight run logs exist, and all eight
runs have written real epoch-progress records. Tracking is local-first because W&B/Hugging Face
credentials are absent. Results persist at
`substrate_rxrx1/cell_dino_cp5/rxrx1_domain_scaling_replicate30_20260810`.

## 2026-08-10 correctness and rigor audit of the active scaling wave

The executable commit `628606c` passed 51 targeted tests in the actual SciServer Python 3.10
runtime, after Python compilation, local regression testing, sweep dry-run, metadata audit, and
run-identity/sharding checks. A programmatic resolved-config comparison confirms that each
quarter/full pair differs only in `train.environment_subset` and its descriptive `run_tag`;
architecture, seed, preprocessing, optimizer, epochs, checkpoint rule, augmentation, losses, and
evaluation code are identical. The quarter subset has all 1,139 labels, all four cell types, and
no outcome-dependent selection. Eight unique start lines were observed and the initial global
fatal scan was empty.

Two limitations are now explicitly frozen rather than hidden:

1. **The current `acc_within` column is not comparable as ID accuracy across scale.** The WILDS
   `id_test` loader contains all 33 original training experiments. At quarter scale, 25 of those
   experiments were not used for training, so calling that aggregate “ID” would be incorrect.
   OOD validation/test are unaffected and remain the primary valid endpoints. The saved terminal
   checkpoints must be re-evaluated on the same eight-experiment `id_test` subset for every arm;
   until then, ignore the active wave's `acc_within` interaction.
2. **Fixed epochs are exposure-matched, not optimizer-step/compute-matched.** Full scale has
   4.12× as many fields and therefore about 4.12× as many updates at 30 epochs. The active wave
   estimates the practical combined effect of more examples, more independent training domains,
   and proportionally more training compute. A positive interaction must be followed by a
   fixed-update control before attributing it specifically to data availability rather than
   optimization compute.

This remains a useful first experiment: its OOD-test interaction directly tests whether the
shared-versus-dense gap changes between a data-poor multi-batch regime and full RxRx1, while
replacement and original arms locate the source of any change. It is a diagnostic at one seed,
not a paper claim. Intermediate points, batch-bootstrap uncertainty, fixed-ID re-evaluation,
fixed-update controls, and fresh-seed replication are required before a scaling conclusion.

The 20-minute automation `rxrx-moe-scaling-research-steward` has been strengthened with mandatory
source, test, resolved-config, data/split, evaluation/artifact, and live-launch correctness gates.
Any failed gate must stop downstream interpretation and launches until repaired and revalidated.

## 2026-08-10 completed seed-5 RxRx1 domain-count scaling protocol

`rxrx1_domain_scaling30_20260810` was a matched 2×4 factorial designed to
answer whether the relative value of sparse adaptation changes with independent batch diversity.
It is explicitly a **training-environment-count curve**, not a generic sample-size curve: adding
experiments simultaneously adds fields and acquisition domains. Site-density and class-count
scaling remain separate axes for later waves.

The low-data point is a stable-hash, cell-stratified prefix of eight of the 33 training
experiments: `[5, 2, 14, 12, 17, 15, 41, 46]`. Metadata audit shows 9,856 training fields, all
1,139 perturbation classes (minimum seven examples per class), and all four cell types. The full
point uses all 33 training experiments and 40,612 fields. OOD validation, OOD test, and ID
evaluation sets are identical across the two points. The label task, native six-channel input,
6→5 Cell-DINO mapping, initialization, optimizer, 30-epoch budget, terminal checkpoint rule,
augmentations, and seed 5 are also fixed.

| Scale | Original Cell-DINO | Dense E4, blocks 10–11 | Replacement E4/top-2 | Shared E3/top-1 |
|---|---|---|---|---|
| 8 train experiments / 9,856 fields | `quarter_original` | `quarter_dense_E4_late2` | `quarter_replace_E4k2_late2` | `quarter_shared_E3k1_late2` |
| 33 train experiments / 40,612 fields | `full_original` | `full_dense_E4_late2` | `full_replace_E4k2_late2` | `full_shared_E3k1_late2` |

This wave estimates the architecture × scale interaction
`(shared − dense at full) − (shared − dense at quarter)` with OOD test accuracy as the headline,
then worst-test-batch accuracy. ID is deferred to the matched eight-environment checkpoint
re-evaluation described above. The replacement arm separates preserving the pretrained dense
path from generic sparse capacity; original separates all expansion from ordinary fine-tuning.
A positive interaction supports the joint data/domain/compute scaling hypothesis even if shared
remains slightly below dense at the current endpoint; the fixed-update follow-up must then isolate
compute. A flat/negative interaction argues that practical scaling under this schedule will not
rescue this MoE design. Subsequent intermediate points and seed replication are gated on this
interaction rather than the best noisy cell.

Implementation adds exact environment-set hashes to run identity, metadata-only manifest audits,
and one-command status/aggregation with test-first paired contrasts. The full local suite passes.
The intended result root is
`substrate_rxrx1/cell_dino_cp5/rxrx1_domain_scaling30_20260810`.

Execution is verified from immutable commit `628606c` in worktree
`/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation-domain-scaling-628606c`.
All four 2xH100 containers each emitted two distinct starts on GPUs 0/1 and later exited cleanly.
The terminal audit found all eight artifacts and no traceback, runtime error, OOM, or `rc=1`.
Tracking was local-first because W&B, Hugging Face, and repository credentials were absent inside
the containers. Results persist at the root above; the completed interaction is summarized at the
top of this file.

## 2026-08-10 causality wave complete — shared MoE is efficient, but does not improve mean test accuracy

`shared_causality30_20260809` is 8/8 terminal. The current project readout is test-first; validation
is retained as secondary diagnostic context. Fresh seeds 3/4 complete the predeclared comparison:

| Seed | Arm | Train | ID | OOD val | OOD test | Worst test batch | Route reliance |
|---:|---|---:|---:|---:|---:|---:|---:|
| 3 | Shared E3/top-1, blocks 10–11 | 100.000% | 55.250% | 21.737% | 38.409% | 9.713% | 0.0497 |
| 3 | Dense E4, blocks 10–11 | 100.000% | 54.511% | 21.829% | 38.418% | 9.426% | — |
| 3 | Shared, balance weight `0` | 100.000% | 55.343% | 21.940% | 38.441% | 9.221% | 0.0471 |
| 3 | Shared, frozen router | 100.000% | 54.841% | 21.342% | 38.136% | 9.016% | 0.0289 |
| 4 | Shared E3/top-1, blocks 10–11 | 100.000% | 54.792% | 21.555% | 38.165% | 8.448% | 0.0457 |
| 4 | Dense E4, blocks 10–11 | 100.000% | 54.981% | 21.788% | 38.781% | 7.295% | — |
| 4 | Shared, balance weight `0` | 100.000% | 54.457% | 21.717% | 38.075% | 8.033% | 0.0401 |
| 4 | Shared, frozen router | 100.000% | 54.691% | 21.717% | 38.026% | 7.910% | 0.0358 |

Across all four clean matched seeds 1–4, canonical shared test accuracy is `38.541%` versus
`38.764%` for dense, a paired mean difference of `-0.223` points. Shared loses the mean-test
comparison in every seed (`-0.223/-0.044/-0.009/-0.616`). Its advantages are instead efficiency
and tail behavior: it activates `4.729M` FFN parameters versus dense's `9.455M`, while averaging
`8.721%` worst-batch accuracy versus `8.125%` for dense (`+0.596`) and `55.214%` ID accuracy versus
`54.856%` (`+0.359`). Secondary validation averages `21.973%` versus `21.798%` (`+0.175`).

The routing variants do not uncover a hidden test gain. Across seeds 1–4, balance-zero averages
`38.494%` test and frozen-router averages `38.371%`, versus `38.541%` for canonical shared. The
single `39.010%` balance-`1e-3` result did not repeat (`38.621%` in its paired seed). No arm reaches
`40%` test. The supported claim is therefore **near-dense test accuracy at about half active FFN
compute, with better average tail accuracy**, not higher RxRx1 test accuracy. This local E3/top-1
late-block routing neighborhood is now sufficiently replicated; another nearby architecture grid
is lower leverage than improving optimization/data scale or making the JUMP-CP substrate real.

## 2026-08-09 routing controls complete — conditional partitioning matters, but the canonical router still wins

`shared_routing30_20260809` is 8/8 terminal. OOD validation remains the decision metric and the
completed canonical E3/top-1 blocks 10–11 model is the same-seed anchor; test is descriptive.

| Seed | Routing arm | Train | ID | OOD val | OOD test | Worst test batch | Route reliance |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Field/image routing | 100.000% | 54.592% | 21.098% | 38.470% | 8.607% | 0.0376 |
| 1 | Balance weight `1e-3` | 100.000% | 55.609% | 22.143% | 39.010% | 9.016% | 0.0462 |
| 1 | Balance weight `0` | 100.000% | 55.220% | 21.910% | 38.787% | 9.180% | 0.0435 |
| 1 | Frozen router during training | 100.000% | 55.338% | 21.900% | 38.566% | 8.238% | 0.0326 |
| 2 | Field/image routing | 100.000% | 53.733% | 22.001% | 38.055% | 8.197% | 0.0397 |
| 2 | Balance weight `1e-3` | 100.000% | 55.065% | 21.768% | 38.621% | 6.762% | 0.0418 |
| 2 | Balance weight `0` | 100.000% | 54.969% | 22.356% | 38.673% | 6.885% | 0.0443 |
| 2 | Frozen router during training | 100.000% | 55.373% | 22.285% | 38.755% | 6.844% | 0.0379 |

Mean OOD validation is `21.550±0.452%` for field routing, `21.956±0.188%` for balance `1e-3`,
`22.133±0.223%` for balance `0`, and `22.093±0.193%` for a frozen router, versus
`22.301±0.046%` for the canonical balance-`1e-2` learned token router. Field routing loses to the
anchor in both seeds (`-1.248/-0.254` points), so per-token conditional routing contains useful
structure. Weaker/no balancing and frozen-router training are mixed across seeds and none beats
the anchor mean. Thus removing the balance term is not a reliable improvement, and gradient-
learning router weights is not consistently necessary at this sample size.

The frozen-router result does not contradict the earlier five-point post-hoc random-route drop.
A fixed input-conditional partition can co-adapt with its experts throughout training; randomizing
routes only after a learned checkpoint breaks that co-adapted partition. The supported mechanism
is therefore **conditional partition/expert co-adaptation**, not yet a claim that the router learns
biologically meaningful semantics. No arm reaches `30%` validation or `40%` test, and the hardest-
batch evidence still does not support general debatching.

### Predeclared fresh-seed causality wave

The next bounded wave is `shared_causality30_20260809`: exactly eight 30-epoch jobs, four matched
arms at fresh seeds 3/4. It directly tests whether the small shared-over-dense effect and the two
ambiguous routing controls survive independent seeds.

| Arm per seed | Question |
|---|---|
| `dense_E4_late2` | Does the shared-residual validation gain continue to beat matched dense widening? |
| `shared_E3k1_late2` | Does the canonical E3/top-1 late-block result replicate again? |
| `shared_E3k1_balance0` | Is removing load balancing reliably neutral/better, or was seed 2 noise? |
| `shared_E3k1_router_frozen` | Can a fixed input-conditional partition match learned routing across seeds? |

This is replication plus causal controls, not another architecture grid. All optimization,
placement, expert count, and active compute are held fixed within the three shared arms; OOD
validation decides every comparison and all test reads are predeclared.

Launch is verified from clean immutable commit `1beae79` after 43 targeted tests passed on
SciServer. Container `2859` runs shard 0 and `2862` runs shard 1, with two processes on GPUs 0/1
in each container. All four seed-3 arms reached epoch 1, all fatal-log scans are empty, and the
four seed-4 arms are queued inside the same healthy controllers. Tracking is local-first because
W&B/HF credentials remain absent. Results persist at
`substrate_rxrx1/cell_dino_cp5/shared_causality30_20260809`.

Seed 3 is now terminal while seed 4 remains in progress. This is an interim readout, not the
predeclared across-seed decision: canonical shared reaches `21.737%` OOD validation, matched
dense `21.829%`, balance-zero shared `21.940%`, and frozen-router shared `21.342%`. Canonical
shared therefore loses narrowly to dense by `0.091` points in this seed, balance zero beats
canonical by `0.203`, and learned routing beats the frozen partition by `0.396`. The corresponding
descriptive test scores are essentially tied (`38.409/38.418/38.441/38.136%`); seed 4 must decide
whether any of these small directions replicate.

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

### Historical prelaunch declaration for the routing-control wave

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

The launch was verified from immutable commit `affb51b` in persistent worktree
`moe-sparse-adaptation-shared-routing-affb51b` after 39 targeted routing/capacity tests passed.
Container `2859` runs shard 0 and `2862` shard 1; each printed two distinct starts on GPUs 0/1.
The campaign later completed 8/8; its terminal evidence is synthesized at the top of this file.
Tracking was local-first because W&B/HF credentials were absent. Results persist at
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
