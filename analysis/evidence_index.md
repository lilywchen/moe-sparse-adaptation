# Evidence and correctness index

Last verified: 2026-07-31 18:24 EDT

This file maps every material claim to its run identity, provenance, split policy, validation status,
and downstream use. `Decision-grade` means eligible for a predeclared scientific decision;
`diagnostic` means useful for debugging or intuition only.

## Active and completed evidence

| Claim or milestone | Grade | Dataset / stage | Runs and persistent artifact | Tested code | Split and test status | Validation / exclusion | Consumer |
|---|---|---|---|---|---|---|---|
| Single-commit seed-0 Phase-A revalidation is running | pending decision-grade | both / Stage 0 | 12-candidate manifest under `hpo_revalidation_26ad7fa/{rxrx1,camelyon17}/phase_a`; 5 active formal workers, 4/12 strictly valid result JSONs | `26ad7fa3b0baa96fae9dab25417e42a844074636` | completed files use `ood_val`; `test_evaluated=false` | 4/6 RxRx1 files passed parseability, finite metrics, filename/config identity, clean commit, parameter-count, and leakage checks; grid incomplete | shared-recipe selection after dataset completion |
| Four RxRx1 Phase-A candidates completed at common provenance | valid but not yet selectable | RxRx1 / Stage 0 | `rxrx1_original_ep30_s0_hpoA_lr1e-04_llrd0.70`; `...lr1e-04_llrd0.85`; `...lr3e-05_llrd0.70`; `...lr3e-05_llrd0.85` under `hpo_revalidation_26ad7fa/rxrx1/phase_a/` | clean `26ad7fa3b0baa96fae9dab25417e42a844074636` | `selection_split=ood_val`; `test_evaluated=false` | 4/4 pass strict schema/provenance/config/finite/parameter checks; excluded from ranking until all 6 cells and sanity gate complete | `analysis/state_update.md`; eventual shared-recipe ranking |
| Earlier Phase-A grid cannot support ranking | diagnostic only | both / Stage 0 | 12 legacy JSONs under the original `hpo/` roots | mixed: `03167d1`, `448c215`, `4795202` | OOD validation only; test not used | excluded from formal ranking because tested commits differ and an environment-ID bookkeeping defect was later repaired | implementation history only |
| Longer DINOv2 training does not by itself resolve weak RxRx1 performance | diagnostic only | RxRx1 / sanity | `hpo/rxrx1/epoch_probe/rxrx1_original_ep90_s0_epochprobe_ep90_lr1e-04_llrd0.70.json`; `...llrd0.85.json` | dirty `4795202` | `selection_split=ood_val`; `test_evaluated=false` | parseable, finite headline metrics, exact config checked; dirty/mixed provenance prevents ranking | `analysis/state_update.md`; RxRx1 sanity gate |
| Canonical WILDS ERM protocol can initialize without OOD-test construction | diagnostic guard | RxRx1 / sanity | `hpo/rxrx1/canonical_erm_dry_6fdfcf1/` | WILDS upstream `4726775` plus compatibility `6fdfcf1` | `evaluate_all_splits=false`; `eval_splits=['id_test']`; no `test_eval.csv` | zero-epoch 0.1% dry-run passed; files limited to train, val, and ID-test | launch gate for canonical reproduction |
| First canonical ERM launch failed before meaningful training | excluded failure | RxRx1 / sanity | `hpo/rxrx1/canonical_erm_sanity_6fdfcf1/launcher.log`; W&B run `26968hby` | `6fdfcf1` | OOD test not constructed or evaluated | excluded explicitly: missing optional `torch_scatter` during group-metric logging | repair audit only |
| Canonical WILDS ERM retry is training | diagnostic, pending result | RxRx1 / sanity | `hpo/rxrx1/canonical_erm_sanity_6fb65e5_retry1/`; W&B run `rytuap3l` | WILDS upstream `4726775` plus audited compatibility `6fb65e5` | validation plus ID-test only; OOD test not constructed/evaluated | native PyTorch group-mean fallback unit-smoked; process alive at epoch 6/batch 565; GPU 0 at 97%; fresh log; no fatal matches; rolling training accuracy 0.1769 is not validation | discriminate substrate/protocol failure from task/data failure |

## Exact diagnostic metrics

| Run ID | OOD-val accuracy | seen-environment accuracy | worst-environment val | Epochs | Eligibility |
|---|---:|---:|---:|---:|---|
| `rxrx1_original_ep90_s0_epochprobe_ep90_lr1e-04_llrd0.70` | 0.0140045 | 0.0944548 | 0.0140045 | 90 | diagnostic only |
| `rxrx1_original_ep90_s0_epochprobe_ep90_lr1e-04_llrd0.85` | 0.0176578 | 0.1041810 | 0.0176578 | 90 | diagnostic only |
| `rxrx1_original_ep30_s0_hpoA_lr1e-04_llrd0.70` | 0.0103511 | 0.0289570 | 0.0008117 | 30 | valid; ranking withheld pending full grid/sanity gate |
| `rxrx1_original_ep30_s0_hpoA_lr1e-04_llrd0.85` | 0.0148163 | 0.0380429 | 0.0020292 | 30 | valid; ranking withheld pending full grid/sanity gate |
| `rxrx1_original_ep30_s0_hpoA_lr3e-05_llrd0.70` | 0.0091333 | 0.0181966 | 0.0012175 | 30 | valid; ranking withheld pending full grid/sanity gate |
| `rxrx1_original_ep30_s0_hpoA_lr3e-05_llrd0.85` | 0.0099452 | 0.0218655 | 0.0008117 | 30 | valid; ranking withheld pending full grid/sanity gate |

## Validation required before a new claim

For every completed CCAS JSON, verify parseability; finite required metrics; exact dataset, seed,
run ID, and config identity; total/active parameter fields; `selection_split=ood_val`;
`test_evaluated=false`; clean common tested commit; and expected manifest membership. Do not count a
file merely because it exists. Do not consume a Stage-1 result without its paired dense control at
the same seed. OOD test remains unavailable until the frozen Stage-3 confirmatory set is recorded.
