# Stage-A Score-Only PWM Sampling (FIMO Log-Odds)

Date: 2026-01-26
Status: accepted

## Context
Stage-A PWM sampling previously relied on p-value strata and retain depth for FIMO-based selection. This created
per-motif tuning overhead and could exclude high-scoring hits when p-value thresholds varied by motif width or
information content. The new requirement is to make Stage-A entirely score-driven using the FIMO/PSSM log-odds
score so that “top retained” always means “highest PWM resemblance.”

## Goals
- Use FIMO log-odds score as the **sole** eligibility/retention signal.
- Ensure deterministic, motif-robust selection across widths and information content.
- Remove all p-value strata/retain-depth configuration and reporting.
- Provide clear score-based UX: recap table, manifest metadata, and a score-based strata overview plot.

## Non-goals
- Introduce new selection knobs or alternative scoring backends.
- Maintain backward compatibility with the prior p-value-based semantics.

## Proposed Changes
### 1) Score-only eligibility, tiering, retention
- For each candidate sequence, compute `best_hit_score` as the **maximum** FIMO log-odds score across forward-strand
  hits. If no hits, candidate is ineligible.
- Eligible iff `best_hit_score > 0`.
- Deduplicate eligible candidates by TFBS sequence before tiering and retention.
- Rank by `(best_hit_score desc, tfbs_sequence asc)` for deterministic tie-breaking.
- Tier by rank with fixed 1% / 9% / 90% counts per regulator.
- Retain top `n_sites` by score; shortfall allowed.

### 2) FIMO invocation
- Always run FIMO with `--thresh 1.0` and `--norc` so the full forward-strand hit set is available.
- Keep background handling explicit: MEME motif background is used (or `bgfile` if provided).

### 3) Config + CLI
- Remove `score_threshold`, `score_percentile`, `pvalue_strata`, and `retain_depth`.
- Keep only: `n_sites`, `oversample_factor`, `mining.batch_size`, `mining.max_seconds`, `mining.log_every_batches`,
  `length_policy`, `length_range`, `scoring_backend: fimo`.
- Add CLI overrides to `dense stage-a build-pool` for `--n-sites`, `--oversample-factor`, `--batch-size`, `--max-seconds`.

### 4) Manifest + outputs
- Pool rows include: `regulator_id`, `tfbs_sequence`, `best_hit_score`, `tier`, `rank_within_regulator` (plus existing
  length/origin/coordinate fields).
- Pool manifest Stage-A metadata includes:
  - tier scheme: `pct_1_9_90`
  - eligibility rule: `best_hit_score > 0 (and has at least one FIMO hit)`
  - retention rule: `top_n_sites_by_best_hit_score`
  - fimo invocation: `thresh=1.0`

### 5) Plotting
- Replace Stage-A strata overview to show:
  - Left panel: eligible score distributions with tier boundary markers and retained overlay.
  - Right panel: retained TFBS length distribution.
- No p-values anywhere in the plot.

## Error Handling
- If Stage-A pools are missing or metadata is incomplete, fail with guidance to rebuild Stage-A pools.
- If FIMO is unavailable, report actionable guidance (use pixi or set MEME_BIN).

## Testing Plan
- Tier assignment determinism and partitioning.
- Retention ordering with deterministic tie-break.
- End-to-end regression for two motifs of different widths:
  - non-zero retained when eligible exists,
  - retained are top scores,
  - plot generation uses `best_hit_score` and tier boundaries.

## Breaking Changes
- Densegen PWM scoring backend removed.
- P-value strata/retain-depth configuration and reporting removed.
