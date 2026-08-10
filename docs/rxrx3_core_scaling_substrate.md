# RxRx3-core paired-experiment scaling substrate

## What the dataset can and cannot test

RxRx3-core is a real, compact six-channel Cell Painting corpus, not a synthetic enlargement of
RxRx1. The public release has 222,601 unique HUVEC wells, 1,335,606 channel images, 1,744
experiment-plate pairs, 736 CRISPR targets, and 1,674 compounds. The images are 512×512 center
crops converted to uint8 and JPEG-2000 compressed. The official image shards are about 17.4 GB;
the current Hugging Face repository is 23.1 GB only because it also contains several embedding
and feature tables that are not required here.

The clean supervised task is **CRISPR gene perturbation identification across paired
experiments**. Query-guide metadata contains 734 target genes: 695 occur in exactly two
experiments, 24 in four, and 15 in only one. Among 176 CRISPR experiments, 174 form 87 exact
pairs with identical query-gene sets. This lets one experiment in each pair be supervised train
and its independently repeated mate be OOD test. The 15 single-experiment genes are excluded.

This differs from RxRx1 in one important way. RxRx1 runs essentially the full label library in
every experiment, while RxRx3-core partitions genes across experiment pairs. Therefore a clean
fixed-label **experiment-count** curve is not identifiable: dropping train experiments also drops
classes. We will not call that data scaling. RxRx3-core instead supports two honest axes:

1. **Plate-count curve:** 1/2/4/8 train plates per train experiment, four CRISPR guides fixed.
   This varies plate-batch diversity and examples per class together.
2. **Guide-count curve:** 1/2/4 guides per gene, eight train plates fixed. This varies biological
   replicate/guide diversity and examples per class while holding plate domains fixed.

## Frozen metadata protocol

`scripts/build_rxrx3_core_gene_manifests.py` applies only outcome-independent metadata rules:

- pair experiments only when their complete query-gene sets are identical;
- retain genes with at least four guide IDs shared across every paired experiment containing the
  gene;
- select four nested guides by QC-retained plate coverage, then stable hash;
- assign one experiment in each pair to train and its mate to OOD test, preferring the member with
  more QC-retained plates for train and using a stable-hash tie break;
- reserve one class-complete train-experiment plate for ID validation;
- require a second class-complete plate so the one-plate training endpoint retains every class;
- use the same ID-validation and OOD-test wells at every point on both curves.

On the official metadata this yields 85 paired train/test experiment groups, 674 fixed gene
classes, 85 train experiments, 85 OOD-test experiments, and zero experiment overlap. Two of the
87 candidate pairs fail the two class-complete-plate rule after guide filtering and are excluded.
The plate curve has 2,696/5,376/10,706/21,404 training wells at 1/2/4/8 plates. Fixed evaluation
contains 2,708 ID-validation wells and 23,855 OOD-test wells, with all 674 classes present in every
split and scale. The guide curve uses the same 85×8 train plates and 1/2/4 nested guides.

Build and hash the manifests with:

```bash
python scripts/build_rxrx3_core_gene_manifests.py \
  --metadata /path/to/metadata_rxrx3_core.csv \
  --output-dir /path/to/rxrx3_core_manifests
```

Generated manifests are licensed data derivatives and must stay outside git. The summary reports
`training_ready: false` until pixels and joins pass the gate below.

## Channels and image gate

RxRx3 uses the same six stains and numeric order as the Recursion datasets: Hoechst, ConA,
Phalloidin, Syto14, MitoTracker, and WGA. Each well has keys ending `_s1_1` through `_s1_6` in
the Hugging Face image Parquet shards. The fixed Cell-DINO map is therefore
`[w1, w2, w4, mean(w3,w6), w5]` = `[DNA, ER, RNA, AGP, Mito]`, applied before per-channel
standardization exactly as in native RxRx1.

Before a GPU launch:

1. Stage only `data/*.parquet`, `metadata_rxrx3_core.csv`, `README.md`, and `LICENSE`; do not
   download the 5.7 GB of unrelated precomputed embeddings/features.
2. Verify all 35 image shards and their exact repository hashes/sizes.
3. Resolve every selected `well_id` to exactly six 512×512 uint8 JP2 rows with channel suffixes
   1–6 and reject duplicate/missing channels.
4. Decode a stratified sample from every shard and verify dimensions, dtype, finite intensity,
   and joint geometric transforms.
5. Keep all raw data, generated manifests, checkpoints, and offline tracking outside git.

Only after this gate should the same four matched Cell-DINO arms—original, dense widening,
replacement sparse MoE, and shared-residual MoE—be predeclared. OOD test and worst-test-
experiment accuracy are headline outcomes; checkpointing uses the fixed ID-validation plates.

## License and provenance

RxRx3-core is distributed under Recursion's current End User License Agreement, not CC0. It
requires the prescribed attribution, restricts several neuroscience/commercial uses, and applies
share-alike-style terms to derivative technology. The paper can report results under those terms,
but checkpoints or other derived artifacts must not be redistributed under an incompatible
license. Exact license text and dataset revision must be archived with the experimental manifest.

- Dataset card: https://huggingface.co/datasets/recursionpharma/rxrx3-core
- Paper: https://arxiv.org/abs/2503.20158
- RxRx3 description and channel stains: https://www.rxrx.ai/rxrx3
