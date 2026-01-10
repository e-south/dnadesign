# Cruncher category/campaign implementation plan (ticketed)

Date: 2026-01-09
Status: Draft execution plan (aligned to `docs/category_campaign_review.md`)

## 0) Purpose

Translate the category/campaign spec into **ticketized, module-level work** with clear acceptance criteria, tests, and file-level scope. This plan is designed to minimize refactors and keep changes decoupled and reproducible.

## 1) Assumptions & constraints

- No core algorithm changes unless explicitly scoped (optimization/scoring stays intact).
- Network access remains explicit and opt-in (fetch + remote inventory commands only).
- All new functionality is additive and must not break existing configs.
- Analysis and campaign summaries operate only on run artifacts.
- No fallbacks: missing/invalid inputs should error clearly.

## 2) Inputs and dependencies

Primary specification:
- `docs/category_campaign_review.md`

Key reference contracts:
- Analysis layout: `src/utils/analysis_layout.py`
- Analysis workflow: `src/workflows/analyze_workflow.py`
- Notebook service: `src/services/notebook_service.py`
- CLI contracts: `docs/cli.md`

## 3) Work breakdown (phased tickets)

### Phase 1 — Config + expansion (low-risk foundation)

**T1. Config schema additions (categories + campaigns)**
- Files:
  - `src/config/schema_v2.py`
- Scope:
  - Add `regulator_categories` and `campaigns` blocks with strict validation.
  - Validate overlaps, size ranges, and selector fields.
- Acceptance criteria:
  - Loading config with new blocks succeeds.
  - Invalid sizes/overlap rules raise a clear error.
  - Existing configs without new blocks still load unchanged.
- Tests:
  - Unit test for validation errors.
  - Unit test for valid minimal config.

**T2. Campaign expansion service**
- Files:
  - `src/services/campaign_service.py` (new)
- Scope:
  - Deterministically expand categories/rules into explicit regulator sets.
  - Dedupe and order results.
- Acceptance criteria:
  - Same input always yields identical output order.
  - Overlap rules behave per spec (distinct vs allowed).
  - Selector filters are applied before combinatorics.
- Tests:
  - Deterministic expansion with fixed input.
  - Overlap + selector edge cases.

**T3. Campaign generate CLI**
- Files:
  - `src/cli/commands/campaign.py` (new)
  - `src/cli/app.py` (register command)
- Scope:
  - `cruncher campaign generate --campaign <name> --out <path>`
  - Write derived config with explicit `regulator_sets`.
  - Write campaign manifest (see spec).
- Acceptance criteria:
  - Generated config loads and runs with existing workflows.
  - Manifest records deterministic expansion and hashes.
- Tests:
  - CLI invoke and output file creation (smoke).

### Phase 2 — Analysis outputs (multi-TF ergonomics)

**T4. Pairwise grid plot (projection matrix)**
- Files:
  - `src/workflows/analyze/plots/summary.py`
  - `src/workflows/analyze/plot_registry.py`
- Scope:
  - Add `score__pairgrid.png` output for N>2 TFs.
  - Register plot in manifest and mark as projection.
- Acceptance criteria:
  - Grid generated for N>2; no crash for N=1/2.
  - Plot registered in `plot_manifest.json`.
- Tests:
  - Minimal integration test with synthetic parquet.

**T5. Joint metrics table**
- Files:
  - `src/workflows/analyze/plots/summary.py`
  - `src/workflows/analyze_workflow.py`
- Scope:
  - Compute and write `analysis/tables/joint_metrics.csv`.
  - Add to `table_manifest.json`.
- Acceptance criteria:
  - Table created for multi-TF runs.
  - Manifest includes new table entry.
- Tests:
  - Unit test for metrics against synthetic score data.

**T6. Notebook surfacing of new outputs**
- Files:
  - `src/services/notebook_service.py`
  - (notebook template string inside `_render_template`)
- Scope:
  - Display `joint_metrics.csv` if present.
  - Show `score__pairgrid.png` in plots tab if available.
- Acceptance criteria:
  - Notebook renders new outputs without breaking strict mode.
- Tests:
  - Snapshot or minimal fixture-based check.

### Phase 3 — Campaign summary (cross-run aggregation)

**T7. Campaign summary workflow**
- Files:
  - `src/workflows/campaign_summary.py` (new)
- Scope:
  - Aggregate run artifacts into `campaign_summary.csv` and `campaign_best.csv`.
  - Generate summary plots in `campaigns/<id>/plots/`.
- Acceptance criteria:
  - Works offline and reads only run artifacts.
  - Produces required columns from spec.
- Tests:
  - Synthetic fixture runs aggregated into correct tables.

**T8. Campaign summarize CLI**
- Files:
  - `src/cli/commands/campaign.py` (extend)
- Scope:
  - `cruncher campaign summarize --campaign <name> --runs ...`.
- Acceptance criteria:
  - Summaries produced; clear error if runs missing artifacts.
- Tests:
  - CLI smoke test.

### Phase 4 — UX polish

**T9. Target listing with category filters**
- Files:
  - `src/cli/commands/targets.py`
- Scope:
  - Add `--category`/`--campaign` filters to list/status/stats/candidates.
- Acceptance criteria:
  - Filtered output matches selected TFs.
  - Existing behavior unchanged without filters.
- Tests:
  - CLI unit test on config with categories.

**T10. Fetch with campaign**
- Files:
  - `src/cli/commands/fetch.py`
- Scope:
  - `cruncher fetch motifs/sites --campaign <name>` resolves TF union.
- Acceptance criteria:
  - Fetch respects campaign-derived TF list.
- Tests:
  - CLI smoke test in offline mode (dry-run).

**T11. Optional campaign-level notebook**
- Files:
  - `src/cli/commands/campaign.py` (optional)
  - `src/services/notebook_service.py` (optional extension)
- Scope:
  - Marimo notebook to explore `campaign_summary.csv`.
- Acceptance criteria:
  - Notebook runs and displays campaign tables/plots.

## 4) Acceptance checklist (system-level)

- [ ] No new implicit network access outside fetch/explicit remote inventory commands.
- [ ] Deterministic campaign expansion (hash-stable).
- [ ] Generated configs are reproducible and load with `load_config`.
- [ ] Analysis outputs register in manifests.
- [ ] Notebook renders new analysis tables/plots.
- [ ] CLI commands fail fast with clear error messages.

## 5) Testing checklist

- Unit tests for schema validation, expansion logic, and joint metrics.
- Integration tests for analyze output registration.
- CLI smoke tests for campaign generate/summarize.

## 6) Deliverables summary (if implemented)

- Category-driven campaigns (deterministic expansion).
- N>2 analysis projections + joint metrics.
- Campaign-level landscape summaries.
- Optional interactive notebooks for exploration.
