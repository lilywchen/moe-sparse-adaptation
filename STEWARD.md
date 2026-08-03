# Autonomous RxRx1 research steward contract

## Fast-first scheduling override (2026-08-03)

Prospectively, long seed-0 training is no longer the default search unit. Preserve healthy jobs
already near a declared checkpoint, but do not restart stalled long-horizon arms or refill a GPU
with another 60/90-epoch recipe unless it passed the fast mechanistic funnel in `PLAN.md`.

Refill priority is now: all-layer gradient-conflict diagnostics from saved checkpoints; two-epoch
learned/frozen/dense mechanism triples at measured conflict peaks and placebo layers; five-epoch
single-versus-multiple sparse-FFN and staged-upcycling screens; then ten-epoch exact pairs. A long
extension requires route reliance of at least one absolute accuracy point, noncollapsed expert
usage, reproducible conflict reduction, and paired accuracy/tail evidence. Training loss and GPU
occupancy never license extension.

The steward executes the RxRx1 kill test in `PLAN.md`; it is not a read-only monitor. It protects
healthy work but uses every safely idle authorized GPU for the next licensed experiment. It does
not launch new Camelyon17 work, the old 36-cell factorial, or additional DINOv2 rescue runs.

The user has subsequently authorized the new Cell-DINO `factorial60` campaign in `PLAN.md`. The
ban on the old natural-image-DINOv2 factorial remains, but it does not prohibit this explicitly
versioned native-CP5 sweep. The steward must keep a longer ready queue than the available GPU pool,
launch disjoint factorial shards across every running container, validate 10/30/60 handoffs, and
immediately refill a GPU released by completion, pruning, or failure. Exploratory winners remain
provisional until their full configuration is locked and paired seeds 1 and 2 replicate it.

## Sources of truth

1. `PLAN.md` — frozen scientific sequence and gates.
2. `PROGRESS.md` — operational ledger and current decision.
3. `analysis/state_update.md` — living scientific interpretation.
4. `analysis/evidence_index.md` — claim-to-run correctness map.
5. `.steward/state.json` and `.steward/action_log.jsonl` — machine state and append-only actions.
6. Persistent SciServer logs/results — execution evidence. PID or W&B status alone is not health.

## Priority order

The steward now operates as an **adaptive signal ladder**. At every result handoff it identifies
the leading live explanations, spends the next bounded batch on experiments that discriminate
between them, and records the stopping/extension rule before launch. A completed negative gate
does not forbid a newly user-authorized scientific question, but previous results are never
relabelled after the fact. Idle authorized GPUs must be filled with independent licensed arms,
validation, or preparation that can change the next decision; they must not be filled with
duplicates or post-hoc variants.

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
7. Preserve the failed canonical replication trigger and route diagnosis as exact-recipe negative
   evidence. Under the later explicit authorization, run the Cell-DINO `factorial60` sweep and
   advance only a locked paired winner to seeds 1–2; do not reinterpret the old gate.
8. For the authorized substrate-strength revision, run the exact ten-arm, 90-epoch hypothesis
   matrix in `PLAN.md` across five disjoint two-GPU shards. Read epoch dependence from the shared
   10/30/60/90 milestones rather than launching separate epoch arms. Interpret capacity, routing,
   representation depth, explicit invariance, and environment weighting as separate contrasts;
   do not replace them with optimizer sweeps. Launch the Channel-Adaptive pair as soon as its
   distinct approved checkpoint passes smoke, and never duplicate or fabricate that instrument.

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
- Never classify a pass as `RUNNING_HEALTHY` merely because one job is alive while another
  authorized GPU is idle and a decision-relevant independent arm is ready. Surface and fill that
  gap in the same handoff.

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
