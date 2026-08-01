# Evidence and correctness index

Last verified: 2026-07-31 23:15 EDT

This file maps every material claim to its run identity, provenance, split policy, validation status,
and downstream use. `Decision-grade` means eligible for a predeclared scientific decision;
`diagnostic` means useful for debugging or intuition only.

## Active and completed evidence

| Claim or milestone | Grade | Dataset / stage | Runs and persistent artifact | Tested code | Split and test status | Validation / exclusion | Consumer |
|---|---|---|---|---|---|---|---|
| Single-commit seed-0 Phase-A revalidation is running | pending decision-grade | both / Stage 0 | 12-candidate manifest under `hpo_revalidation_26ad7fa/{rxrx1,camelyon17}/phase_a`; 3 active Camelyon17 workers, 9/12 strictly valid result JSONs | `26ad7fa3b0baa96fae9dab25417e42a844074636` | completed files use `ood_val`; `test_evaluated=false` | all 6 RxRx1 files and 3 Camelyon17 files passed parseability, finite metrics, filename/config identity, clean commit, parameter-count, and leakage checks | Camelyon17 shared-recipe selection after its grid completes; RxRx1 grid excluded by competence gate |
| Six RxRx1 Phase-A candidates completed at common provenance | valid and ranked; selection gated | RxRx1 / Stage 0 | all six `rxrx1_original_ep30_s0_hpoA_*` JSONs under `hpo_revalidation_26ad7fa/rxrx1/phase_a/` | clean `26ad7fa3b0baa96fae9dab25417e42a844074636` | `selection_split=ood_val`; `test_evaluated=false` | 6/6 pass strict schema/provenance/config/finite/parameter checks; formal rank recorded, but recipe freeze is withheld because dense-substrate sanity fails | `analysis/state_update.md`; RxRx1 go/no-go gate |
| Earlier Phase-A grid cannot support ranking | diagnostic only | both / Stage 0 | 12 legacy JSONs under the original `hpo/` roots | mixed: `03167d1`, `448c215`, `4795202` | OOD validation only; test not used | excluded from formal ranking because tested commits differ and an environment-ID bookkeeping defect was later repaired | implementation history only |
| Longer DINOv2 training does not by itself resolve weak RxRx1 performance | diagnostic only | RxRx1 / sanity | `hpo/rxrx1/epoch_probe/rxrx1_original_ep90_s0_epochprobe_ep90_lr1e-04_llrd0.70.json`; `...llrd0.85.json` | dirty `4795202` | `selection_split=ood_val`; `test_evaluated=false` | parseable, finite headline metrics, exact config checked; dirty/mixed provenance prevents ranking | `analysis/state_update.md`; RxRx1 sanity gate |
| Canonical WILDS ERM protocol can initialize without OOD-test construction | diagnostic guard | RxRx1 / sanity | `hpo/rxrx1/canonical_erm_dry_6fdfcf1/` | WILDS upstream `4726775` plus compatibility `6fdfcf1` | `evaluate_all_splits=false`; `eval_splits=['id_test']`; no `test_eval.csv` | zero-epoch 0.1% dry-run passed; files limited to train, val, and ID-test | launch gate for canonical reproduction |
| First canonical ERM launch failed before meaningful training | excluded failure | RxRx1 / sanity | `hpo/rxrx1/canonical_erm_sanity_6fdfcf1/launcher.log`; W&B run `26968hby` | `6fdfcf1` | OOD test not constructed or evaluated | excluded explicitly: missing optional `torch_scatter` during group-metric logging | repair audit only |
| Canonical WILDS ERM retry demonstrates learnability | diagnostic, decision-relevant sanity evidence | RxRx1 / sanity | `hpo/rxrx1/canonical_erm_sanity_6fb65e5_retry1/`; W&B run `rytuap3l`; evaluation CSVs through epoch 47 | WILDS upstream `4726775` plus audited compatibility `6fb65e5` | validation plus ID-test only; OOD test not constructed/evaluated | epoch 47 best-so-far OOD-val 0.154151; epoch-21 train/ID-test/OOD-val 0.7075/0.2458/0.1341; exact validation files inspected | establishes dataset learnability and fixed competence reference |
| Complete bounded DINOv2 rescue fails competence gate | decision-grade substrate exclusion | RxRx1 / competence gate | four JSONs under `hpo/rxrx1/dense_rescue_26ad7fa/` and `hpo/rxrx1/dense_rescue_1da67a5/` | first three clean `26ad7fa3b0baa96fae9dab25417e42a844074636`; official transform GitHub `aa8d0cf` applied as clean SciServer `1da67a5` | all use `selection_split=ood_val`, `test_evaluated=false`; OOD test untouched | 4/4 pass parseability, finite metrics, exact identity/config, clean or code-equivalent provenance, equal 21,628,800 parameters, and split/test guards; best rescue 0.055003 is 35.7% of canonical 0.154151, below frozen 50% gate | excludes natural-image DINOv2 from RxRx1 factorial; licenses replacement-backbone decision, not extra tuning |
| Final Camelyon17 Phase-A shard launched | pending decision-grade | Camelyon17 / Stage 0 | `camelyon17_original_ep10_s0_hpoA_lr3e-04_llrd0.85`; persistent phase-A log; W&B `fwab3bqs` | clean `26ad7fa3b0baa96fae9dab25417e42a844074636` | selection configured on `ood_val`; startup states test untouched | exact sole-pending shard dry-run; no duplicate/result; GPU0 free; verified process, 98% GPU utilization, fresh log, W&B, clean checkout | completes parallel coverage of all remaining Camelyon17 candidates |
| Cell-DINO implementation/checkpoint are valid; linear and `1e-4` full-FT results are valid | validated implementation; 2/3 diagnostic results valid | RxRx1 / competence gate | GitHub `db9ebdb`; SciServer checkout `moe-sparse-adaptation-run-db9ebdb-v2`; checkpoint `models/cell_dino/cell_dino_vits8_pretrain_cp-37d20e9c.pth`; valid `kill_rxrx1/competence/rxrx1_original_ep5_s0_linear_probe.json` and `...full_ft_lr1e-4.json`; active `...full_ft_lr3e-4` | GitHub and SciServer tree `c537d33ef4db3dc2206c0922a0159b4b19095adc`; SciServer `4c1a0ab2` (results record unique short `4c1a0ab`); DINOv2 `7764ea0`; checkpoint SHA-256 `37d20e9cd48b3d610b5de15a4ea4e7e060a593b8d8358e928d079dc7b03ee66a` | train/ID/OOD validation only; both valid results have `selection_split=ood_val`, `test_evaluated=false`; OOD test untouched | 80/80 tests; safe load and real-checkpoint smoke; both JSONs pass parseability, finite metrics, exact identity/config/provenance and split guards; `3e-4` active at epoch 5 with fresh log | licenses completion/validation of the final full-FT diagnostic, not MoE comparison |
| Frozen Cell-DINO non-parametric readouts launched | diagnostic pending | RxRx1 / representation diagnosis | `scripts/probe_rxrx1_cell_dino_oob.py`; planned result `kill_rxrx1/oob/rxrx1_cell_dino_frozen_oob_readouts_s0.json`; isolated checkout `moe-sparse-adaptation-run-oob-601ff00` | base SciServer `4c1a0ab2`; uploaded script SHA-256 `2541cdff100e139222225d5c617965eac54e81bba7ba895ceb7e641287ecd33b` | extracts train embeddings and evaluates ID-test/OOD-validation only; result hard-codes `selection_split=ood_val`, `test_evaluated=false`; OOD-test loader never iterated | 4 focused tests and 82/82 full suite pass; no duplicate/result; GPU0 free; exact process, 70% utilization/7.1 GiB, fresh log, and split sizes verified | distinguishes weak perturbation geometry from experiment-local/batch-confounded geometry before MoE |

## Exact diagnostic metrics

| Run ID | OOD-val accuracy | seen-environment accuracy | worst-environment val | Epochs | Eligibility |
|---|---:|---:|---:|---:|---|
| `rxrx1_original_ep90_s0_epochprobe_ep90_lr1e-04_llrd0.70` | 0.0140045 | 0.0944548 | 0.0140045 | 90 | diagnostic only |
| `rxrx1_original_ep90_s0_epochprobe_ep90_lr1e-04_llrd0.85` | 0.0176578 | 0.1041810 | 0.0176578 | 90 | diagnostic only |
| `rxrx1_original_ep30_s0_hpoA_lr1e-04_llrd0.70` | 0.0103511 | 0.0289570 | 0.0008117 | 30 | valid; ranked, selection gated by sanity |
| `rxrx1_original_ep30_s0_hpoA_lr1e-04_llrd0.85` | 0.0148163 | 0.0380429 | 0.0020292 | 30 | valid; ranked, selection gated by sanity |
| `rxrx1_original_ep30_s0_hpoA_lr3e-05_llrd0.70` | 0.0091333 | 0.0181966 | 0.0012175 | 30 | valid; ranked, selection gated by sanity |
| `rxrx1_original_ep30_s0_hpoA_lr3e-05_llrd0.85` | 0.0099452 | 0.0218655 | 0.0008117 | 30 | valid; ranked, selection gated by sanity |
| `rxrx1_original_ep30_s0_hpoA_lr3e-04_llrd0.70` | 0.0102496 | 0.0246725 | 0.0008117 | 30 | valid; ranked, selection gated by sanity |
| `rxrx1_original_ep30_s0_hpoA_lr3e-04_llrd0.85` | 0.0133956 | 0.0448143 | 0.0008117 | 30 | valid; ranked, selection gated by sanity |
| `rxrx1_original_ep30_s0_rxdiag_no_rrc` | 0.0154252 | 0.0451344 | 0.0012175 | 30 | valid diagnostic only; clean `26ad7fa` |
| `rxrx1_original_ep30_s0_rxdiag_uniform_lr` | 0.0132941 | 0.0367379 | 0.0008117 | 30 | valid diagnostic only; clean `26ad7fa` |
| `rxrx1_original_ep30_s0_rxdiag_no_rrc_uniform_lr` | 0.0185711 | 0.0522998 | 0.0020292 | 30 | valid diagnostic only; clean `26ad7fa` |
| `rxrx1_original_ep30_s0_rxdiag_wilds_uniform_lr` | 0.0550030 | 0.1052891 | 0.0101461 | 30 | valid diagnostic; clean `1da67a5`; best rescue but fails gate |
| `rxrx1_original_ep5_s0_linear_probe` | 0.0287193 | 0.0404806 | 0.0097403 | 5 | valid Cell-DINO diagnostic; frozen backbone; not a MoE comparison |
| `rxrx1_original_ep10_s0_full_ft_lr1e-4` | 0.0484067 | 0.0859352 | 0.0137987 | 10 | valid Cell-DINO diagnostic; full fine-tuning; competence gate incomplete |

## Validation required before a new claim

For every completed CCAS JSON, verify parseability; finite required metrics; exact dataset, seed,
run ID, and config identity; total/active parameter fields; `selection_split=ood_val`;
`test_evaluated=false`; clean common tested commit; and expected manifest membership. Do not count a
file merely because it exists. Do not consume a Stage-1 result without its paired dense control at
the same seed. OOD test remains unavailable until the frozen Stage-3 confirmatory set is recorded.
