# Sparse Conditional Adaptation Under Acquisition Shift

This repository contains the code and manuscript for a controlled study of one question:

> When adapting a pretrained vision model to scientific images collected across acquisition
> environments, when does sparse conditional capacity improve held-out-environment generalization
> over an equally large dense model?

The study uses DINOv2 ViT-S/14, RxRx1, and Camelyon17. One FFN block is upcycled at a time. Every
primary MoE comparison uses the nearest realizable total-parameter-matched dense FFN, identical
data, optimization, placement, seed, and (where applicable) output-invariance objective.

## Study in one table

| Factor | Levels |
|---|---|
| Placement | early, middle, late |
| Routing unit | image, token |
| Router geometry | linear, cosine |
| Training pressure | canonical LBL, within-environment LBL, output invariance |

This is a 3 × 2 × 2 × 3 = 36-cell MoE factorial per dataset. Dense controls are run at every
placement under canonical training and output invariance. The route-level arm shares the
canonical dense control because a dense model has no routing distribution to balance.

## Repository map

- `paper/`: compilable ICLR manuscript.
- `configs/`: frozen study configurations.
- `moe_shift/capacity/`: parameter-matched dense and MoE upcycling.
- `scripts/run_ccas.py`: one auditable run.
- `scripts/sweep_ccas.py`: resumable factorial sweep.
- `scripts/aggregate_ccas.py`: paired contrasts, factorial effects, and stage-gated reporting.
- `scripts/sweep_rxrx1_shared_performance.py`: eight-run shared/residual-MoE performance wave,
  sharded two jobs per 2×H100 container.
- `scripts/summarize_rxrx1_performance_wave.py`: one-command live validation/test result table.
- `tests/`: protocol and analysis guards.
- `docs/research_plan.html`: detailed operational research plan.
- `SCISERVER.md`: cluster paths and safe execution/check-in contract.
- `PROGRESS.md`: shared handoff ledger for humans and scheduled Codex checks.

## Local validation

```bash
python -m pip install -e '.[test]'
pytest -q
python scripts/sweep_ccas.py --dry-run --dataset rxrx1 --seeds 0
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

PyTorch and torchvision are intentionally not installed by the package: SciServer should keep its
CUDA-matched builds. See `SCISERVER.md` before launching any run.

## Stage gate

Stage 1 and model selection use only WILDS OOD validation. The OOD test split is not evaluated by
the runner until `stage=3`. A configuration is eligible for confirmation only if the dense/MoE
budget audit passes and the selection rule in `PLAN.md` has been applied without test access.

The current ICLR style files are the latest official release available as of July 2026 (ICLR
2026). Replace them with the official ICLR 2027 files when released; do not edit the style file.

## Shared/residual-MoE performance wave

The bounded performance wave consists of six standard MoE allocation/top-k/depth arms, one
cross-experiment supervised-contrastive arm, and one MixStyle arm. All eight candidates are
predefined and receive the same terminal OOD-test readout; OOD validation remains the only model
selection metric. Launch container `i` with:

```bash
python scripts/sweep_rxrx1_shared_performance.py \
  --shard-index i --num-shards 4 --gpus 0,1 --max-concurrent 2
```

At any time, print the complete global table with:

```bash
python scripts/sweep_rxrx1_shared_performance.py --status
```
