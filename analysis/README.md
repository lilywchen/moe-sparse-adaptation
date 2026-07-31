# Analysis artifacts

The steward writes the current evidence synthesis to `analysis/state_update.md` after each
material experimental milestone. `analysis/evidence_index.md` maps claims to exact runs, paths,
commits, split policy, audits, exclusions, and downstream tables or manuscript sections. Generated
tables and figures belong in subdirectories here; large raw results and checkpoints remain in
SciServer persistent storage.

All pre-confirmatory analysis must use OOD validation only. The aggregate script's stage gate must
remain enabled, and no OOD-test value may enter this directory or the manuscript before Stage 3.
