# Cruncher MMR + Config Simplification Design (2026-02-04)

## Intent

Cruncher should produce reproducible, high-scoring, diverse TF-binding sequences with assertive, no-fallback behavior. This design simplifies configuration, removes ambiguous knobs, and hardens fixed-length behavior while aligning MMR selection with the objective (consensus-like per TF).

## Scope

In scope:
- Remove trim/polish and length auto-opt to enforce fixed-length runs.
- Make MMR the only elite selection policy (canonical, TFBS-core, tolerant weights).
- PT-only optimization in config/CLI (Gibbs hidden for now).
- Schema hard-breaks with explicit errors for removed keys.
- Tests and docs aligned with new behavior.

Out of scope:
- Changing MCMC acceptance kernels.
- New objective functions or spacing penalties.
- Deep performance refactors beyond safe reuse/caching notes.

## MMR Behavior (Document Explicitly)

MMR compares candidate sequences across the elite pool, not motif-to-motif within the same sequence:
- For each sequence in the candidate pool, extract the best-hit window for each TF (e.g., LexA core, CpxR core), orient to PWM.
- When comparing two sequences, compute LexA-core vs LexA-core and CpxR-core vs CpxR-core weighted Hamming distances, then average across TFs.
- Never compare LexA vs CpxR within the same sequence.

## Diversity Weights (Hard-Coded)

Use TFBS-core weighted distance with **tolerant** weights:
- weight = 1 - info_norm per PWM position.
- This preserves consensus-critical positions and encourages diversity where motifs are flexible.

## Config Changes (Hard Break)

Remove:
- `sample.output.trim.*`
- `sample.output.polish.*`
- `sample.auto_opt.length.*`
- `sample.elites.min_hamming`
- `sample.elites.dsDNA_hamming`
- `sample.elites.selection.distance.*`
- `sample.elites.selection.relevance_norm`
- `optimizer.name = gibbs`

Enforce:
- `sample.init.length >= max_pwm_width` (schema + runtime)
- MMR canonicalization when `objective.bidirectional=true`

## Behavior Changes

- PT is the only optimizer selectable via config/CLI.
- No warm-start or auto-disable fallbacks for removed features.
- Fixed-length is explicit: trimming/polishing are not available.

## Refactors

- Split `app/sample/auto_opt.py` into candidate generation, scoring/selection, and orchestration.
- Split `app/sample/run_set.py` into run layout, candidate pool construction, MMR selection/metadata, and manifest writing.

## Tests

- Config validation tests reject removed keys.
- CLI smoke test: minimal two-TF demo run with tiny budgets.
- Fixed-length validation test (length < max PWM width hard errors).

## Docs

Update:
- `docs/guides/sampling_and_analysis.md`
- `docs/reference/config.md`

Include:
- Fixed-length requirement and removal of trim/polish/length auto-opt.
- MMR TFBS-core behavior and tolerant-weight explanation.
- PT-only optimizer selection.

## Performance Note

Profiling shows PWM DP table construction dominates short runs. Consider safe scorer reuse across auto-opt pilots in a later pass.
