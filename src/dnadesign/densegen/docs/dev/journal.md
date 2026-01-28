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
- Audit follow-up: fixed resume crash from numpy array truthiness in attempts parsing; allow outputs/ root for failure counts; removed run_metrics boolean reindex warning; fixed positional occupancy plot column selection; Stage-B build-libraries now emits sampling-pressure events; demo doc notes --overwrite for libraries; added tests for failure counts, required columns, and Stage-B events.
- Audit follow-up: Stage-A strata overview now accepts regulator_id/tfbs_sequence columns and expands length axis for long TFBS; Stage-B build-libraries error suggests --overwrite; demo doc snippet uses uv/pixi python; added tests for Stage-A plot columns/length axis and Stage-B overwrite hint.

### 2026-01-27
- Task: plot refactor to scalable canonical set (placement_map, tfbs_usage, run_health, stage_a_summary, stage_b_summary) and delete per-subsample plots.
- Changes: plot registry + plotting overhaul, new Stage-A yield/bias panels, Stage-B feasibility/utilization summary, updated demo configs/docs + tests; added Stage-B feasibility fields to library_builds and Stage-A yield counters in pool manifest.
- Tests: updated plot registry/manifest and run diagnostics tests; fixed run_health tick labeling warning; adjusted Stage-A summary expectations.
- Follow-up: placement_map now accepts effective_config.json structure (nested config), with test coverage; demo run uses pixi with -c path due to pixi running from repo root.
- Follow-up: enabled core-unique sampling defaults (unique_binding_cores=true), added tfbs_core to binding_sites inputs, surfaced dedupe_by/min_core_hamming_distance in pool manifest, and added screen-style progress dashboard with show_tfbs/show_solutions toggles; updated demo/config/docs/tests to schema 2.6 and added core-uniqueness tests.
- Follow-up: split pipeline helpers into core/pipeline modules (inputs/progress/attempts/versioning), added CLI path display `--absolute` and PWM sampling progress toggles; added tests for inspect run path rendering and progress toggling (commit a99b2d5).
- Follow-up: modularized pipeline further with core/pipeline/outputs.py and core/pipeline/stage_b.py, corrected relative imports, and updated tests to import new modules; adjusted resume test to expect Stage-A pool staleness error and added show_tfbs/show_solutions args to _process_plan_for_source callers.
- Follow-up: enforced workspace-relative path rendering for CLI errors and workspace init, added CLI test for missing-config path display, and updated docs to reflect Stage-A caching semantics and new inspect/run flags.
- Follow-up: Stage-B CLI now writes feasibility fields (fixed_bp/min_required_bp/slack_bp/infeasible/sequence_length) using shared feasibility helper; inspect run library summaries are aggregated (no per-library breakdown) with short TF labels by default; `dense plot` output paths are workspace-relative by default; run gating ignores events.jsonl so Stage-A/Stage-B prep doesn’t block `dense run`; updated CLI/docs/tests to remove per-library flags.
- Follow-up: added motif display name helper module, switched CLI/reporting/pipeline logs to short TF labels by default, and introduced workspace-relative path helper; updated tests for inspect run library summary (hide TFBS sequences) and Stage-B build-libraries summary (no per-library rows); refined library summary aggregation in reports. Commit: bb36164.
- Follow-up: Stage-B build-libraries CLI output now aggregates per input/plan (min/med/max), removes per-library rows and show-hash flag; docs updated to match; demo note added.
- Follow-up: updated inspect run library CLI test to cover --show-tfbs flag after removing --top; full pytest passes.

### 2026-01-28
- Task: Implement Stage-A "score + diversity" selection policy (MMR), core-identity collapse via matched sequence, adaptive tier-target mining, schema bump to 2.7 with deprecation rewrite, update pool manifest/parquet metadata, CLI recap, tests, and docs.
- Decisions:
  - Uniqueness defaults to core for PWM inputs; min_core_hamming_distance deprecated/ignored with warning.
  - Tier targeting via mining.budget.mode=tier_target with required_unique tracking + explicit warnings when unmet.
  - Selection policy defaults to top_score; MMR optional with alpha=0.9 and shortlist controls.
  - No Stage-B changes.
- Follow-up: updated demo configs and docs to new Stage-A sampling schema (mining budget, length/selection/uniqueness blocks), refreshed CLI Stage-A plan/recap columns for tier-target and selection policy, and added config migration test coverage.
- Follow-up: centralized Stage-A/Stage-B sampling narrative into docs/guide/sampling.md, replaced demo prose with links, added placement_map fixed-element note, and fixed MMR tier-widening selection to avoid empty shortlist crashes with a regression test.
- Follow-up: fixed fixed_candidates mining to stop at the requested budget (prevented long-running FIMO tests), updated mining shortfall/core dedupe tests, and confirmed full densegen test suite passes.
- Task: Stage-A summary visual redesign (publication-quality layout).
- Decisions: dedicated header row via GridSpec, yield/dedupe funnel plot, anchored tier summary box, shared axes cleanup, and Okabe-Ito color mapping with lexA/cpxR overrides.
