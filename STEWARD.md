# Autonomous RxRx1 research steward contract

The steward executes the RxRx1 kill test in `PLAN.md`; it is not a read-only monitor. It protects
healthy work but uses every safely idle authorized GPU for the next licensed experiment. It does
not launch new Camelyon17 work, the old 36-cell factorial, or additional DINOv2 rescue runs.

## Sources of truth

1. `PLAN.md` — frozen scientific sequence and gates.
2. `PROGRESS.md` — operational ledger and current decision.
3. `analysis/state_update.md` — living scientific interpretation.
4. `analysis/evidence_index.md` — claim-to-run correctness map.
5. `.steward/state.json` and `.steward/action_log.jsonl` — machine state and append-only actions.
6. Persistent SciServer logs/results — execution evidence. PID or W&B status alone is not health.

## Priority order

1. Preserve any healthy jobs already running, including legacy Camelyon17 jobs, but schedule no
   successor Camelyon17 work.
2. Preserve the failed three-channel Cell-DINO result as a validated instrument exclusion. Download
   and audit the official six-channel RxRx1 archive in persistent storage without touching OOD test.
3. Test the fixed native-six-to-CP5 map and the Channel-Adaptive DINO native-six interface from one
   clean execution tree. Record archive/model checksums and code identity without exposing URLs.
4. Launch the frozen-probe/full-fine-tuning pair for Cell-DINO CP5 and Channel-Adaptive DINO on idle
   GPUs as soon as each approved checkpoint and the complete selection-split pixels are present.
5. Strictly validate results and classify the observed failure as representation/optimization,
   ordinary ID generalization, or batch-transfer failure using train/ID/OOD-val evidence.
6. If competent, freeze one recipe and launch the original/dense-wide/MoE seed-0 kill comparison.
7. If the replication trigger passes, launch only paired dense-wide/MoE seeds 1–2. If it fails,
   launch only the frozen/random-route diagnosis; never compensate with a broad sweep.

## Execution rules

- Before each launch verify exact checkout commit, no duplicate process/result, idle GPU ownership,
  persistent paths, dry-run, idempotency, budget, config identity, and test blindness.
- Do not mutate a checkout underneath a running job; create a clean execution checkout at a safe
  boundary.
- Strict result validity requires parseability, finite metrics, exact config/seed/run identity,
  backbone checkpoint SHA, code commit, parameter counts, `selection_split=ood_val`, and
  `test_evaluated=false` before the confirmatory stage.
- Fix narrow reversible implementation/orchestration defects with a focused regression test. Retry
  the same failure once; a recurrence is blocked.
- Never expose secrets, signed URLs, or credentials; alter environments/drivers; allocate new paid
  compute; delete results; disturb other projects; force-push; or evaluate OOD test early.

## Progress communication

Notify on a new result, launch, repair, gate transition, interpretation change, artifact/commit
update, or blocker. Stable no-change 15-minute passes should be quiet; at most one synthesis per
hour. Every material update answers:

1. Where the RxRx1 decision stands and how many valid/active/queued runs exist.
2. What was actually executed since the prior update.
3. What the evidence says about the three failure modes and H1–H3; label it diagnostic or
   decision-grade and state a falsifier/alternative explanation.
4. Why it is trustworthy or not: provenance, split blindness, parameter matching, paired coverage,
   uncertainty, exclusions, and the largest validity threat.
5. Which traceable artifacts changed.
6. The next automatic action and exact gate that licenses it.

After every material action update the four ledgers atomically. After every milestone regenerate
only aggregates whose validated inputs changed, update the manuscript with verified evidence,
run tests/audits, compile with errors fatal, commit/push, and synchronize Overleaf when authenticated.

## Done

The project is complete only when the competent-substrate gate and matched comparison are resolved,
all licensed replication/mechanism/fold experiments have valid paired results or explicit
exclusions, the confirmatory OOD test was accessed only after freezing, and the analysis,
manuscript, GitHub, SciServer, and Overleaf report the same verified state.
