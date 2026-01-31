--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/docs/plans/2026-01-31-stage-a-tier-ladder-score-norm-reporting-design.md

Design for unifying Stage-A tier ladders and reporting score_norm summaries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------

# Stage-A tier ladder + score_norm reporting (design)

Date: 2026-01-31

## Context
Stage-A currently has two ladders: diagnostic score tiers and the MMR pool tier_widening ladder. This duplicates
config, adds knobs, and creates user confusion when “tier fill” does not match MMR pool selection behavior.
We also want to report score strength relative to the theoretical max (score_norm) without using it as a pool gate.

## Goals
- Canonicalize ladder configuration so “tier fill” and MMR pool selection are driven by the same fractions.
- Remove min_score_norm as a pool gate; keep it as report-only.
- Add score_norm min/median/max summaries for top vs diversified candidates in recap + manifest.
- Keep MMR behavior intact: greedy Carbonell/Goldstein with weighted Hamming similarity.
- Avoid extra compute during live mining progress (summary-only reporting).

## Non-goals
- No changes to Stage-B.
- No additional live/interactive progress metrics.
- No new backward compatibility for removed ladder config (clean break).

## Design

### Configuration
- Introduce or use `sampling.tier_fractions` as the single canonical ladder for Stage-A tiers and MMR pool slicing.
- Remove `selection.tier_widening` entirely. If present, fail fast with a clear config error.
- Keep `selection.pool.min_score_norm` but treat it as **report-only** (no filtering).
- Keep `selection.pool.max_candidates` as explicit compute cap only.

### Pool construction
- After unique dedup, rank eligible_unique by best_hit_score.
- Choose smallest tier fraction `f` from `sampling.tier_fractions` whose slice size ≥ n_sites.
- MMR pool = top ceil(f * eligible_unique). No min_score_norm gating.
- If pool ≤ n_sites, skip MMR (degenerate) and return top-score order; record in summary/manifest.

### Reporting
- Compute `score_norm = best_hit_score / pwm_theoretical_max_score` for reporting only.
- Recap table: add a compact column with top vs diversified score_norm min/median/max.
- Manifest: store structured fields for top/diversified score_norm min/median/max, plus optional per-tier
  score_norm summaries for eligible pool (tier0/1/2/rest).

## Data flow
1) Mining produces candidates + best_hit_score.
2) Dedup by uniqueness.key.
3) Rank eligible_unique by score.
4) Select tier slice using canonical tier_fractions.
5) Apply optional pool.max_candidates cap.
6) Run MMR or mark degenerate if pool ≤ n_sites.
7) Compute score_norm summaries for top/diversified and per-tier eligible summaries.

## Error handling
- Fail fast if `selection.tier_widening` is present.
- Fail fast if `sampling.tier_fractions` is missing or invalid.
- If `pwm_theoretical_max_score` <= 0, raise a clear error before reporting score_norm.

## Testing
- Unit tests for ladder unification (pool selection uses tier_fractions, no tier_widening accepted).
- Summary tests for score_norm min/median/max fields in recap + manifest.
- Degenerate pool test ensures MMR is skipped and labeled appropriately.

## Migration
- Update docs and examples to use `sampling.tier_fractions` only.
- Remove any references to `selection.tier_widening`.

## Open questions
- None for this design. Implementation should proceed with a clean break from `selection.tier_widening`.
