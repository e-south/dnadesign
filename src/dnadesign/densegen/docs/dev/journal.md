## Journal

### 2026-01-26
- Task: Stage-A PWM sampling overhaul to FIMO score-only semantics; remove p-value strata and p-value-based retention.
- User decision: FIMO-only backend; remove densegen backend and score_threshold/score_percentile knobs.
- Tie-breaker for equal scores: tfbs_sequence lexicographic ascending.
- Branch: working on densegen/cruncher-refine (stay on this branch).
- Spec notes:
  - Use FIMO log-odds score only; best_hit_score = max score per candidate (forward strand only).
  - Eligibility: has at least one hit and best_hit_score > 0.
  - Deduplicate by tfbs sequence before tiering and retention.
  - Tiering per regulator: 1% / 9% / 90% by score rank with deterministic counts.
  - Retention per regulator: top n_sites by score (tie-break by sequence); shortfall allowed.
  - FIMO run must use thresh=1.0 so score-only eligibility works across motifs.
  - Remove p-value strata/retain-depth knobs and p-value-based plots/summary.
- User decision: drop mocks across densegen tests; convert to real FIMO-backed tests and skip when FIMO unavailable.
- Commit timing: commit once the mock-removal work is done; include new untracked files but keep the stray Excel temp file untracked.
- Follow-up: removed unused Stage-A sampling vars after ruff failures; ruff check/format now clean.
- Follow-up: rank_within_regulator is now 1-based; added end-to-end Stage-A FIMO test and score-tier helper module; docs/plot registry updated to score-only semantics.
- Follow-up: centralized FIMO report threshold, removed p-value-only fields from FIMO hits, added non-finite score guardrails, and documented 1-based ranks in outputs.
- Follow-up: Stage-A manifest now surfaces bgfile/background_source and FIMO threshold lives in core constants; legacy score_percentile examples removed from docs.
- Audit: ran full demo flow; Stage-B build-libraries failed when required_regulators used short labels not matching PWM motif IDs.
- Fixes: demo config now uses motif IDs (lexA_CTGTATAWAWWHACA, cpxR_MANWWHTTTAM) and demo docs call out label source.
- Added Stage-B CLI error handling for missing regulators + improved core error message to include available regulators; new CLI test added.
- Docs: clarified required_regulators must match Stage-A pool tf labels; outputs reference now mentions bgfile/background_source in Stage-A metadata.
- Plan note: proceed with ROI diagnostics plots; create run_metrics in outputs/tables; standardize resample/stall markers in outputs/meta/events.jsonl.
- Follow-up: implemented run_metrics aggregation + pipeline write; added run-level diagnostics plots (timeline funnel, failure pareto, library health, score traceability, offered vs used) and plot runner support; updated outputs/docs/demo to document run_metrics and diagnostics plots; added tests for run_metrics and plot functions.
- Follow-up: added sampling pressure events + run_metrics aggregation, failure breakdown by library, library slack distribution, positional occupancy plot with fixed-element overlay, and dense_arrays placement fallback for traceability; updated docs/demo and added tests.
