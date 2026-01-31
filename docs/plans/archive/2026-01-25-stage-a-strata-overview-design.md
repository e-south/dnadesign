# Stage-A Sampling UX + Strata Overview Plot

Date: 2026-01-25
Status: accepted

Note: This plan is historical and includes legacy p-value-based semantics. The current
Stage-A behavior is score-only and documented in src/dnadesign/densegen/docs/guide/sampling.md.

## Context
Stage-A PWM sampling (FIMO backend) already records per-regulator p-value strata and retention, but the CLI output and plots are hard to interpret. Users need immediate, low-noise feedback on the configured strata/retain depth before sampling begins, and a canonical visualization that shows eligible-vs-retained behavior without relying on debug logs. Existing plots (`stage_a_pvalue_strat_hist`, `stage_a_length_hist`) are partial, redundant, and not aligned with the desired narrative.

## Goals
- Show **per-regulator** `pvalue_strata` and `retain_depth` *before* sampling starts.
- Make Stage-A recap focus on **signal**: candidates → eligible → retained, plus per-bin counts.
- Provide **one canonical composite plot** per input: eligible p-value distribution + retained length distribution.
- Keep plots **decoupled** from debug logs (`keep_all_candidates_debug`).
- Fail fast on missing inputs/metadata (no fallbacks).

## Non-goals
- Change the sampling algorithm or selection logic.
- Add new selection modes beyond stratified top-N for FIMO.
- Provide automatic recovery if Stage-A metadata is missing.

## Proposed Changes
### 1) Stage-A plan block (pre-run)
Add a “Stage-A plan” block in `dense stage-a build-pool` that prints **one row per regulator** before sampling starts. It includes:
- input name
- regulator (motif ID)
- backend
- p-value strata edges
- retain depth (and implied floor)

Implementation detail: enumerate motif IDs from PWM sources (MEME/JASPAR/artifact/CSV) at plan time, even when `motif_ids` is not specified (scan headers to list all).

### 2) Stage-A recap table (post-run)
Reduce noise and align terms with semantics:
- `candidates` = generated/target
- `eligible` = unique hits at/below floor (generated denominator)
- `retained` = retained/n_sites (top-N within retained bins)
- `bins` = eligible/retained per bin
- `len(n/min/med/avg/max)` = retained pool lengths

Input name appears as a section header (not repeated per row) unless mixed backends require it.

### 3) Persist eligible bin counts in pool_manifest
Extend `outputs/pools/pool_manifest.json` with a `stage_a_sampling` block per input:
- `backend`
- `pvalue_strata` (edges)
- `retain_depth`
- `eligible_bins`: list of `{regulator, counts}` aligned with `pvalue_strata`

This enables plots without candidate logs and avoids re-deriving eligibility later.

### 4) Canonical composite plot
Add a new plot (name TBD, e.g., `stage_a_strata_overview`) that produces **one figure per input**:
- **Left panel:** p-value **step lines** per regulator (eligible counts by bin). X = -log10(p), Y = eligible count. Add vertical dashed line at retain-depth boundary.
- **Right panel:** retained length histogram per regulator (same colors).

Plot inputs: Stage-A pool parquet + `pool_manifest.json` `stage_a_sampling` block. If required metadata is missing, error with guidance to rebuild Stage-A pools.

### 5) Remove redundant plots
Remove `stage_a_pvalue_strat_hist` and `stage_a_length_hist` from registry and docs. Configs referencing them will error (intentional; no backward compatibility).

## Data Flow / Ownership
- Sampling (`pwm_sampling.py`) computes per-regulator eligible/retained bin counts.
- Pool artifact build writes summary + new `stage_a_sampling` manifest data.
- CLI recap uses sampling summaries for concise reporting.
- Plot reads only `outputs/pools/*` + `pool_manifest.json` (decoupled from debug logs).

## Error Handling
- If Stage-A pools are missing: fail with “run stage-a build-pool first.”
- If `stage_a_sampling` metadata missing: fail with “rebuild pools with --fresh.”
- If input backend lacks strata (non-FIMO): plot is not available and fails fast if requested.

## Testing Plan
- Update plot registry tests to expect the new plot and ensure manifest entries exist.
- Add tests verifying `pool_manifest.json` contains `stage_a_sampling` for FIMO inputs.
- Add plotting tests for: missing metadata, missing pools, successful composite output.
- Update CLI tests for the new Stage-A plan block + recap formatting.

## Docs / Demo Updates
- Update plot list and demo flow to call the new composite plot after `stage-a build-pool`.
- Remove references to deprecated Stage-A plots.

## Breaking Changes
- Remove old Stage-A plots (no compatibility alias). Update any configs that reference them.
