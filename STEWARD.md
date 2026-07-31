# Autonomous research steward contract

The scheduled steward is an **executor**, not a read-only monitor. Its job is to advance the
predeclared study from the current durable state to the next verified milestone and eventually to
a complete analysis and synchronized manuscript.

## Sources of truth

1. `PLAN.md` defines the scientific protocol and stage gates.
2. `PROGRESS.md` records the human-readable current state and next safe action.
3. `.steward/state.json` records the machine-readable heartbeat signature and last action.
4. `.steward/action_log.jsonl` is the append-only evidence/action ledger.
5. SciServer's persistent results and logs are authoritative for remote execution state. A PID or
   W&B run alone is never proof that a job is alive.

When these disagree, investigate and repair the ledger; do not guess.

## One scheduled invocation

Each invocation must first classify the project into exactly one state:

- **RUNNING_HEALTHY:** the expected MoE launcher/children exist, GPU utilization and recent log or
  checkpoint progress agree, and there is no fatal error. Emit a compact heartbeat. Do not launch
  duplicate work, edit code gratuitously, or consume another GPU.
- **ACTIONABLE:** no healthy expected job is running and the next action is already licensed by
  `PLAN.md`. Continue doing useful work until a healthy next GPU job is verified running, or—when
  the next milestone is analytical—until the analysis, progress ledger, manuscript, GitHub, and
  Overleaf are updated and verified.
- **BLOCKED:** progress requires credentials, a destructive action, a scientific-design change,
  additional compute authorization, resolution of a repeated failure, or permission to cross an
  explicit protocol gate. Preserve state and report the exact evidence and smallest user decision.
- **COMPLETE:** all authorized stages, analysis, manuscript synthesis, repository synchronization,
  and verification criteria below are complete. Keep a lightweight completion heartbeat; do not
  invent more experiments.

An invocation must not stop after merely observing that a previous job ended. It owns the handoff:
validate outputs, aggregate them, fix bounded operational problems, update state, and launch the
next predeclared work when the gates pass.

## Work policy

- Prefer the largest safe bounded unit of work that can be verified within one invocation.
- Fix narrow, reversible implementation or orchestration defects; add a regression test; rerun
  only missing idempotent work. After the same failure recurs, stop as `BLOCKED` rather than loop.
- Do not change hypotheses, factors, levels, outcomes, selection rules, fairness constraints, or
  the OOD-test gate. A scientific change requires the user.
- Never cancel a healthy job or touch another project's files, processes, W&B runs, or GPUs.
- Update `.steward/state.json`, append `.steward/action_log.jsonl`, and update `PROGRESS.md` after
  every material action. Every entry includes timestamp, evidence, action, result, commit, and next
  gate.

## Analysis and paper handoff

At every completed experimental milestone:

1. Run the stage-appropriate leakage, coverage, budget, pairing, and provenance audits.
2. Regenerate the stage-gated aggregate artifacts under the persistent results directory.
3. Update `analysis/state_update.md` with facts only: completed coverage, failures/exclusions,
   parameter audit, primary paired contrasts, uncertainty, factor effects, mechanism findings,
   dataset agreement/disagreement, and unresolved limitations. Never expose OOD-test values before
   the Stage-3 gate.
4. Update manuscript results/methods/limitations and generated tables or figures. Clearly retain
   placeholders for evidence that does not yet exist.
5. Run tests and compile `paper/main.tex` locally. Commit and push the scoped changes to GitHub.
6. Sync the linked Overleaf project, recompile there, and verify that the linked GitHub commit,
   selected main document, and PDF build all succeeded. Record verification in `PROGRESS.md`.

Linked Overleaf project:
`https://www.overleaf.com/project/6a6cdb522d6aa17eed95038d`

## Definition of done

- Every run required by the authorized frozen protocol has a valid, provenance-complete result or
  a documented exclusion.
- Required paired controls, seed coverage, budget matching, and stage-gate audits pass.
- The analysis artifacts and manuscript faithfully report the completed evidence, including null
  results and cross-dataset disagreement.
- Tests pass; the manuscript builds locally and on Overleaf.
- Local `main`, GitHub `main`, the SciServer execution commit, and Overleaf are synchronized to the
  same reported research state.

