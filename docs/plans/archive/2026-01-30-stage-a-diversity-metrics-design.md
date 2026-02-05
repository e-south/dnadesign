# Stage-A Diversity Metrics + Plot Redesign


## Contents
- [Context](#context)
- [Goals](#goals)
- [Non-goals](#non-goals)
- [Decisions](#decisions)
- [Metric definitions](#metric-definitions)
- [Plot changes](#plot-changes)
- [Testing plan](#testing-plan)
- [Docs updates](#docs-updates)

## Context
Stage-A diversity diagnostics currently show minimal change when MMR is score-dominated, but the plots and
metrics make that harder to read. We need a proof-oriented, low-noise summary that aligns with the actual
MMR objective and provides a clean “final diversity outcome” view.

## Goals
- Use score normalization based on PWM max (consensus log-odds) for objective reporting.
- Compute exact pairwise distances for retained sets to avoid sampling noise.
- Report MMR objective gain (ΔJ) and diversity gain (Δdiv) with clear definitions.
- Replace the entropy panel with a score‑vs‑diversity contribution plot that reflects MMR’s selection logic.
- Keep plots interpretable without extra categorical labels or knobs.

## Non-goals
- Change Stage-A selection behavior or the MMR algorithm.
- Add new configuration knobs or user-tuned thresholds.
- Modify Stage-B sampling or solver behavior.

## Decisions
- Normalize scores using PWM max (consensus log-odds). Store `pwm_max_score` and use it for
  `score_norm = best_hit_score / pwm_max_score` in Stage-A summaries.
- Use exact pairwise weighted‑Hamming distances (PWM‑tolerant weights) for retained sets.
- Compute ΔJ with selection‑order nearest‑distance (mirrors MMR objective) and Δdiv using
  median pairwise distance (final outcome view).
- Diversity plot uses pairwise ECDF (Top‑score vs MMR) for outcome, plus score‑vs‑diversity
  contribution (MMR selection‑time nearest distance) on the right panel.

## Metric definitions
- ΔJ (objective gain): J(MMR) − J(Top‑score), using normalized scores and selection‑order similarity
  (same distance model as MMR).
- Δdiv (distance gain): median(pairwise distance, MMR) − median(pairwise distance, Top‑score),
  computed from exact pairwise distances on `tfbs_core`.

## Plot changes
- Diversity plot left: ECDF of exact pairwise distance (Top‑score vs MMR), explicit legend handles.
- Diversity plot right: score_norm vs selection‑time nearest distance (MMR contribution).
- Remove entropy panel and redundant annotations; keep outcome-first view.

## Testing plan
- Unit tests for new summary fields (`pwm_max_score`, score_norm quantiles, ΔJ, Δdiv).
- Unit tests for exact pairwise summaries (n_pairs equals total pairs, subsampled false).
- Plot tests updated for new labels and axes.

## Docs updates
- Sampling guide and demo: define ΔJ/Δdiv and score normalization basis.
- Outputs reference: add new fields in `pool_manifest.json` and plot semantics.
