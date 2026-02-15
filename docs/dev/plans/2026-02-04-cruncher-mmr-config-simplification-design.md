# Cruncher MMR + Config Simplification Design (2026-02-04)

## Contents
- [Intent](#intent)
- [Scope](#scope)
- [MMR behavior](#mmr-behavior)
- [Diversity weights](#diversity-weights)
- [Config invariants](#config-invariants)
- [Behavior](#behavior)
- [Tests](#tests)
- [Docs](#docs)
- [Performance note](#performance-note)

## Intent

Cruncher produces reproducible, high-scoring, diverse TF-binding sequences with assertive, no-fallback behavior. This design keeps configuration minimal, enforces fixed-length sampling, and aligns MMR selection with the objective (consensus-like per TF).

## Scope

In scope:
- Fixed-length sampling with explicit validation.
- MMR as the canonical elite selection policy (TFBS-core, tolerant weights).
- Parallel tempering as the optimizer surface in config/CLI.
- Schema hard errors for unsupported keys.
- Tests and docs aligned with current behavior.

Out of scope:
- Changing MCMC acceptance kernels.
- New objective functions or spacing penalties.
- Deep performance refactors beyond safe reuse/caching notes.

## MMR behavior

MMR compares candidate sequences across the elite pool, not motif-to-motif within the same sequence:
- For each sequence in the candidate pool, extract the best-hit window for each TF (e.g., LexA core, CpxR core) and orient to the PWM.
- When comparing two sequences, compute same-TF core distances (LexA vs LexA, CpxR vs CpxR) and average across TFs.
- Different TFs are never compared within the same sequence.

## Diversity weights

TFBS-core distance uses tolerant weights:
- weight = 1 - info_norm per PWM position.
- This preserves consensus-critical positions and encourages diversity where motifs are flexible.

## Config invariants

- `sample.init.length` is enforced and must be >= the widest PWM length.
- Elite selection uses MMR with TFBS-core distance and deterministic tie-breaks.
- Bidirectional scoring implies dsDNA canonicalization for uniqueness and elite selection.
- Parallel tempering is the only optimizer exposed via configuration.

## Behavior

- No warm-start or auto-disable behavior is used.
- Fixed-length sampling is explicit and consistent across runs.
- Auto-opt remains a PT pilot selector with deterministic scorecard tie-breaks.

## Tests

- Config validation tests reject unsupported keys.
- CLI smoke test for a minimal two-TF demo run with tiny budgets.
- Fixed-length validation test (length < max PWM width hard errors).

## Docs

Update:
- `src/dnadesign/cruncher/docs/guides/sampling_and_analysis.md`
- `src/dnadesign/cruncher/docs/reference/config.md`

Include:
- Fixed-length requirements.
- MMR TFBS-core behavior and tolerant-weight explanation.
- PT-only optimizer behavior.

## Performance note

Profiling shows PWM DP table construction dominates short runs. Consider safe scorer reuse across auto-opt pilots in a later pass.
