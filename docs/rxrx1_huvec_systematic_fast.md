# RxRx1 HUVEC systematic fast study

This campaign is a one-seed screening study over the official six-channel HUVEC images. It is
designed to establish whether independently measured experiment distance predicts perturbation
classification difficulty, then test a matched dense/MoE pair only after both a ResNet and dense
ViT pass explicit training gates.

For a non-specialist visual explanation of experiments, wells, sites, channels, custom splits,
Cell-DINO difficulty, preliminary evidence, and the full-data certification gate, open
[`rxrx1_huvec_study_explainer.html`](rxrx1_huvec_study_explainer.html).

## Persistent root

```text
/home/idies/workspace/Storage/lchen5/persistent/moe-sparse-adaptation/
substrate_rxrx1/huvec_systematic_fast_20260814
```

The result root contains the frozen site manifest, six Cell-DINO/QC cache shards, split registry,
probe results, raw-model logs/results/checkpoints/predictions, and the final analysis report.

## Automatic stages

1. Validate all native channel files and freeze the 24-experiment, 1,108-treatment manifest.
2. Extract frozen Cell-DINO features and raw-QC features on all six H100s.
3. Freeze three experiment folds and controlled low/medium/high source resamples.
4. Run centroid and linear probes and render the initial embedding/QC figures.
5. Require ResNet-18 and ViT-Tiny to memorize a fixed canary set.
6. Run ResNet confirmation and certify the full-source dense ViT.
7. Run the matched dense/MoE primary and controlled comparisons.
8. If mean primary MoE gain is positive, run the total-parameter-matched dense control.
9. Run dense and MoE masked-autoencoder pretraining on the hardest frozen controlled split.
10. Aggregate all predictions into CSV tables, figures, and `analysis/REPORT.md`.

Targets are excluded from normalization, training, source-IID validation, checkpoint selection,
and masked-autoencoder pretraining. Training uses individual sites; evaluation averages the two
site logits into one well prediction.

## Three-container launch

Run one command in each 2×H100 container, varying only `--shard-index` from 0 to 2:

```bash
python scripts/sweep_rxrx1_huvec_study.py \
  --shard-index 0 --num-shards 3 --gpus 0,1 --max-concurrent 2
```

The launch is resumable. Existing terminal results are skipped, failed subprocesses write a
persistent failure record, and all launchers stop if any shared failure appears.

## Live status and final report

```bash
python scripts/sweep_rxrx1_huvec_study.py --status
```

The final human-readable entry point is:

```text
analysis/REPORT.md
```
