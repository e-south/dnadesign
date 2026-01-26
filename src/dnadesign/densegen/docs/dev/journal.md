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
