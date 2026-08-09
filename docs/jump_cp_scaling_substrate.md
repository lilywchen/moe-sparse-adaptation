# JUMP-CP cross-source scaling substrate

## Purpose

This is the larger-data branch for testing whether shared-residual sparse upcycling becomes more
useful as supervised Cell Painting adaptation scales. It is not a larger copy of RxRx1. The task
is **compound perturbation identification from Cell Painting fields**, with acquisition source held
out as the domain shift.

The design uses the public JUMP Cell Painting Consortium `cpg0016-jump` corpus. Official metadata
has 115,786 non-control compound identities across 738,406 wells. The fixed five-source
intersection below contains **11,423 non-control compounds**, roughly ten times the RxRx1 label
space, with the same compound identity observed in every selected acquisition source.

## Frozen protocol

| Role | JUMP acquisition source |
|---|---|
| Supervised train domains | `source_1`, `source_2`, `source_3` |
| OOD validation domain | `source_8` |
| OOD test domain | `source_10` |

- Label: `Metadata_JCP2022` compound identity.
- Controls: every ID in `perturbation_control.csv` is excluded.
- Class ladders: 1,024, 4,096, and all 11,423 common compounds, selected once by a stable hash.
- Initial balanced unit: one deterministic well per class per source and one image site per well.
  This gives 5,120, 20,480, and 57,115 fields respectively once a site is resolved.
- Selection: OOD validation only. OOD test is a descriptive readout for frozen arms.
- Channels presented to Cell-DINO: `[DNA, ER, RNA, AGP, Mito]`. The physical file columns must be
  mapped by stain semantics from the load-data index; numeric channel position must not be assumed.
- Primary comparison at every ladder: original Cell-DINO, matched dense expansion, replacement
  MoE, and shared-residual MoE under the same optimization protocol and seeds.

The manifest builder is metadata-only and deterministic:

```bash
python scripts/build_jump_compound_cross_source_manifest.py \
  --metadata-root /path/to/jump-datasets/metadata \
  --output-dir /path/to/manifests
```

It writes source/plate/well rows plus hashes and explicitly reports `training_ready: false`. This
prevents a metadata manifest from being mistaken for a resolved image dataset.

## Acquisition gate

The full Cell Painting Gallery is much too large to mirror casually: the official registry reports
roughly 126.8 TB of raw images for this collection. The one-site balanced all-class subset is
currently estimated around 0.9 TB from corpus-wide averages, but that is not an allocation number.
It must be replaced by an exact unsigned-S3 object listing and byte sum.

Before downloading pixels:

1. Fetch only the official load-data/image index for the five selected sources.
2. Join it to the frozen source/plate/well manifests and select one deterministic site per well.
3. Validate all five stain paths, dimensions, bit depth, and channel semantics on a small sample.
4. Produce exact object and byte counts for each class ladder.
5. Verify SciServer storage headroom, then stage the smallest ladder first with checksums.

No JUMP image index or raw pixels are currently present in the verified SciServer inventory, so no
scaling run is authorized yet. The blocker is concrete acquisition/storage validation—not model
code or label definition.

## Provenance and license

- Corpus and data description: [JUMP Hub](https://broadinstitute.github.io/jump_hub/explanations/data_description.html)
- Public object registry: [Cell Painting Gallery on AWS](https://registry.opendata.aws/cellpainting-gallery/)
- Canonical metadata: [jump-cellpainting/datasets](https://github.com/jump-cellpainting/datasets)

Cell Painting Gallery data are released under CC0 1.0 with publication citation requested. The
metadata repository is BSD-3-Clause. Record exact metadata commit and file hashes with every built
manifest.
