# Cruncher category/campaign specification (ergonomics + extensibility)

Date: 2026-01-09
Status: Implemented (phases 1-4 complete; demo expanded)

## 0) Purpose

Define a **pragmatic, decoupled, and extensible** path to support regulator *categories* and *multi-regulator (N>2) campaigns* in Cruncher, with a UX that scales beyond pairwise workflows. This document is a **granular, implementation-ready specification** aligned to current architecture, artifacts, and CLI contracts.

## 1) Scope

In-scope:
- Category-aware selection and expansion into explicit `regulator_sets`.
- Campaign-level UX for “run many combinations” and compare results.
- Multi-regulator analysis/EDA: pairwise grids, joint metrics, and succinct summaries.
- Deterministic, reproducible artifacts; offline analysis; no changes to core sampling correctness.

Out-of-scope (explicitly unchanged):
- Core scoring/optimization algorithms and their numerical semantics.
- Lockfile schema and verification behavior.
- Existing parse/sample/analyze/report stage semantics (only additive extensions).

## 2) Principles (must-haves)

- **Decoupled:** No network in compute/analysis; network access is explicit in fetch and remote inventory commands. Campaign expansion is offline.
- **Reproducible:** Generated configs/manifests are deterministic and versionable.
- **Pragmatic:** Prefer additive fields and helper commands to large refactors.
- **Assertive:** Validate inputs early; fail clearly for ambiguous categories, duplicates, or invalid rules.
- **Extensible:** Schema and CLI designed for future category rules without breaking existing configs.
- **Minimal redundancy:** Reuse existing run artifacts and analysis layout; avoid duplicating per-run logic.

## 3) Alignment with current Cruncher architecture (critical constraints)

This spec is grounded in existing architecture and artifact contracts:

### 3.1 Lifecycle and decoupling
- Fetch is the only network stage; analyze/report are offline. (`docs/architecture.md`, `docs/spec.md`)
- Core compute is I/O-free. (`docs/architecture.md: core/`)

### 3.2 Regulator grouping
- `cruncher.regulator_sets` is the **only** runtime grouping mechanism and already supports N>2 TFs.
  - `src/config/schema_v2.py:CruncherConfig.regulator_sets`
  - `src/workflows/sample_workflow.py:_run_sample_for_set`

### 3.3 Analysis layout
- Analyze writes `analysis/summary.json`, `analysis/plot_manifest.json`, `analysis/table_manifest.json`, and `analysis/tables/*.csv`.
  - `src/workflows/analyze_workflow.py:run_analyze`
  - `src/utils/analysis_layout.py:ANALYSIS_DIR_NAME, TABLES_DIR_NAME`

### 3.4 Notebook integration
- `cruncher notebook` already generates a marimo notebook for a run, validated against `summary.json` and `plot_manifest.json`.
  - `docs/cli.md: cruncher notebook`
  - `src/cli/commands/notebook.py`
  - `src/services/notebook_service.py`

### 3.5 Pairwise limitation
- Pairwise plots require `analysis.tf_pair` (exactly 2 TFs). (`src/config/schema_v2.py:AnalysisConfig._check_tf_pair`, `src/workflows/analyze/plot_registry.py:PLOT_SPECS`)

This spec **extends** these contracts; it does not replace them.

## 4) Terminology

- **Regulator (TF):** a named PWM target (e.g., LexA).
- **Category:** named set of regulators (can overlap).
- **Regulator set:** explicit list of TFs used in a single run.
- **Campaign:** rules that expand categories into multiple regulator sets.
- **Selector:** filter on candidate TFs (info bits, site count, dataset preference, etc.).
- **Objective rule (optional):** how TF scores are combined during optimization (future phase).

## 5) Requirements

### R1: Category + campaign expansion
- Users define **categories** and **campaign rules** in config.
- Expansion is deterministic and produces explicit `regulator_sets`.
- Overlaps are handled explicitly (`allow_overlap` / `distinct_across_categories`).

### R2: Campaign ergonomics
- Users can **generate** a derived config and inspect expanded sets.
- Users can **fetch** all TFs implied by a campaign.
- Users can **summarize** many runs with a single command (landscape view).

### R3: Multi-regulator analysis
- Pairwise plots are treated as **projections** of the N-D space.
- Provide a **pairwise grid** for all TF interactions.
- Provide **joint metrics** to quantify how “jointly optimized” sequences are.

### R4: Quality metrics
- Compute PWM quality metrics (info bits, site counts, dataset provenance) as stable metadata.
- Expose these metrics in summaries and selection rules, without altering optimization.

### R5: UX for category selection
- Users can request: “pick one TF from category A and one from B.”
- Users can request: “optimize for at least one TF from category A and one from B.”
- Overlapping TFs must not allow trivial one-TF optimization to satisfy multiple categories.

## 6) Proposed configuration (additive)

### 6.1 Schema additions (conceptual)

```yaml
cruncher:
  regulator_categories:
    <CategoryName>: [TF_A, TF_B, ...]

  campaigns:
    - name: <campaign_name>
      categories: [Category1, Category2, ...]
      within_category:
        sizes: [2, 3]                 # combos within a single category
      across_categories:
        sizes: [2, 3]                 # combos across categories
        max_per_category: 2
      allow_overlap: true
      distinct_across_categories: true
      dedupe_sets: true
      selectors:
        min_info_bits: 8.0
        min_site_count: 10
        source_preference: [regulondb, coldbacteria]
        dataset_preference: [dataset_1, dataset_2]
      tags:
        organism: ecoli
        purpose: multi_tf_sweep
```

### 6.2 Example for provided categories

```yaml
cruncher:
  out_dir: runs/

  regulator_categories:
    Category1: [CpxR, BaeR]
    Category2: [LexA, RcdA, Lrp, Fur]
    Category3: [Fnr, Fur, AcrR, SoxR, SoxS, Lrp]

  campaigns:
    - name: regulators_v1
      categories: [Category1, Category2, Category3]
      within_category:
        sizes: [2, 3]
      across_categories:
        sizes: [2, 3]
        max_per_category: 2
      allow_overlap: true
      distinct_across_categories: true
      dedupe_sets: true
      selectors:
        min_info_bits: 8.0
        min_site_count: 10
      tags:
        organism: ecoli
        purpose: multi_tf_sweep

  regulator_sets: []  # optional explicit sets
```

## 7) Directory-level change map (module-level spec)

This is the minimal set of modules to touch, aligned to existing architecture:

- `src/config/schema_v2.py`
  - Add `regulator_categories` and `campaigns` config blocks.
  - Add validation for overlap rules, size ranges, and selectors.

- `src/config/load.py`
  - Load new fields; no side effects beyond validation.
  - Expansion remains in a **service**, not the config loader.

- `src/services/campaign_service.py` (new)
  - Expand categories + rules into explicit regulator sets.
  - Deterministic ordering + dedupe.
  - No I/O beyond reading config payload.

- `src/cli/commands/campaign.py` (new)
  - `campaign generate`: write derived config + campaign manifest.
  - `campaign summarize`: aggregate run artifacts into cross-run tables/plots.

- `src/cli/commands/targets.py`
  - Add `--category` and `--campaign` filters for list/status/candidates/stats.

- `src/workflows/analyze/plots/summary.py`
  - Add pairwise grid plot.
  - Add joint metrics table (new CSV).

- `src/workflows/analyze/plot_registry.py`
  - Register pairwise grid plot (and label it “projection”).

- `src/workflows/analyze_workflow.py`
  - Add new table outputs to `analysis/tables` and `table_manifest.json`.
  - Keep existing analysis artifacts unchanged.

- `src/workflows/campaign_summary.py` (new)
  - Offline aggregation of multiple runs into summary tables + plots.

- `src/services/notebook_service.py`
  - Extend run-level notebook to surface new tables/plots if present.

- `tests/*`
  - Add unit tests for campaign expansion, validation, and summary outputs.

## 8) Data contracts & artifacts

### 8.1 Campaign manifest (new)

Path (example):
- `<workspace>/.cruncher/campaigns/<campaign_id>.json` or alongside generated config.

Required fields:
- `campaign_id` (stable hash of config + rules + selectors)
- `campaign_name`
- `created_at`
- `source_config` (path + sha256)
- `categories` (resolved map)
- `selectors` (resolved)
- `rules` (sizes, overlap, dedupe)
- `expanded_sets` (list of TF lists, deterministic order)
- `expanded_count`

### 8.2 Generated config (derived)

Path: specified via `--out`.

Required behavior:
- Must include explicit `cruncher.regulator_sets` matching `expanded_sets`.
- Should include a `campaign` metadata block (readable by CLI) but **not** required by core workflows.

### 8.3 Analysis tables/plots (per-run)

Existing analysis layout (must preserve):
- `analysis/summary.json`
- `analysis/plot_manifest.json`
- `analysis/table_manifest.json`
- `analysis/tables/score_summary.csv` (existing)
- `analysis/tables/elite_topk.csv` (existing)

Additions (new):
- `analysis/tables/joint_metrics.csv`
  - Columns: tf_names, joint_min, joint_mean, joint_hmean, balance_index, pareto_front_size, pareto_fraction
- `analysis/plots/score__pairgrid.png`
  - Pairwise grid; labeled as projection.

Table manifest updates:
- Register `joint_metrics.csv` in `analysis/table_manifest.json`.

Plot manifest updates:
- Register `score__pairgrid.png` with description “pairwise projection grid.”

### 8.4 Campaign summary outputs (cross-run)

Output layout:

```
<out_dir>/campaigns/<campaign_id>/
  campaign_manifest.json
  campaign_summary.csv
  campaign_best.csv
  plots/
    best_jointscore_bar.png
    tf_coverage_heatmap.png
    pairgrid_overview.png
```

Required columns for `campaign_summary.csv`:
- run_name, set_index, tf_list, n_tfs
- n_sequences, n_elites
- joint_min_best, joint_mean_best
- balance_index_best
- per-TF mean/median scores
- PWM quality stats (info bits, site counts)

## 9) CLI UX specification (additive)

### 9.1 Generate explicit regulator_sets

```
cruncher campaign generate --campaign regulators_v1 --out config.generated.yaml
```

Behavior:
- Expands campaign rules into explicit `regulator_sets`.
- Writes campaign manifest.
- No network calls.

### 9.2 Fetch all campaign targets

```
cruncher fetch motifs --campaign regulators_v1 <config>
cruncher fetch sites  --campaign regulators_v1 <config>
```

Behavior:
- Resolves union of TFs implied by campaign.
- Reuses existing fetch logic per TF.

### 9.3 Summarize a campaign (landscape view)

```
cruncher campaign summarize --campaign regulators_v1 --runs runs/*
```

Behavior:
- Aggregates per-run artifacts into cross-run tables and plots.
- No network calls.

### 9.4 Category selection previews

```
cruncher targets list --category Category2
cruncher targets stats --category Category2
cruncher targets candidates --category Category2 --fuzzy
```

### 9.5 Notebook UX (existing command, extended)

- `cruncher notebook` already generates a marimo notebook per run.
- Extend it to show `joint_metrics.csv` and `score__pairgrid.png` if present.
- `cruncher campaign notebook` explores `campaign_summary.csv` outputs.

## 10) UX semantics for category selection

### 10.1 Selection vs objective

Define two distinct concepts:
- **Selection rule**: which TFs are included in a regulator set.
- **Objective rule**: how TF scores are combined during optimization.

### 10.2 Supported selection modes (phase 1)

- **Cross-category combinations**: choose 1 TF from each category, or any size specified.
- **Within-category combinations**: choose k TFs within a single category.
- **Selectors** filter candidates before combinations (min info bits, site count, dataset preference).

### 10.3 Objective semantics (phase 2, optional)

To support “at least one TF from category A and one from category B” within a single run:
- Compute a **category score** = `max` (or softmax) of TF scores within that category.
- Combine category scores via `min` or `sum` (configurable).
- If a TF appears in multiple categories, treat it **once** to avoid double counting.

This is a deeper change (scoring), so it is optional and staged later.

## 11) Analysis & visualization specification

### 11.1 Pairwise grid (N>2 view)

Purpose: show all pairwise projections for N-D optimization.

- Input: `sequences.parquet` score columns for all TFs.
- Output: `analysis/plots/score__pairgrid.png` (optional PDF).
- Use subsampling to cap point counts.

### 11.2 Joint optimization summaries

Per run, compute and persist:
- `joint_min`, `joint_mean`, `joint_hmean` (across TF scores).
- `balance_index = min / mean` (higher is more balanced).
- `pareto_front_size` and `pareto_fraction` (elites).
- Optional `hypervolume` (if implemented with a fixed reference point).

Output:
- `analysis/tables/joint_metrics.csv`.

### 11.3 Category satisfaction metrics

If categories defined, compute:
- `% sequences where all categories have ≥1 TF above threshold`.
- `% sequences where every TF exceeds threshold`.
- `% sequences where at least one TF per category exceeds threshold`.

### 11.4 Existing plots remain, with clarity updates

- Pairwise scatter plots must be labeled as **projections**.
- Parallel coordinates should highlight the joint score (color scale) to spot balanced elites.

## 12) Quality metrics and selection rules

### 12.1 Quality signals (existing sources)

- PWM info bits: `src/core/pwm.py:PWM.information_bits`.
- Site counts and length stats: `src/services/target_service.py:target_stats`.
- Provenance tags: `src/store/catalog_index.py:CatalogEntry`.

### 12.2 Where quality scoring lives

- Catalog-level metrics: compute once (ingest/catalog), reused across campaigns.
- Run-level metrics: derived from run artifacts in analysis/campaign summary.

## 13) Validation & error handling

Explicit errors (no fallbacks):
- Missing category name in campaign rules.
- Invalid size ranges or `max_per_category`.
- Overlap conflicts when `distinct_across_categories=true`.
- Selector filters produce zero candidates.
- Generated config missing or invalid `regulator_sets`.

## 14) Implementation checklist (phased)

### Phase 1 — Config + expansion (low risk)
- [x] Add `regulator_categories` + `campaigns` to schema.
- [x] Implement `campaign_service.expand_campaign()`.
- [x] CLI `cruncher campaign generate`.
- [ ] Optional: record campaign metadata into run manifest extras.

### Phase 2 — Analysis extensions
- [x] Add pairwise grid plot to analysis outputs.
- [x] Add `joint_metrics.csv` and register in table manifest.
- [x] Update notebook scaffold to surface new outputs.

### Phase 3 — Campaign summary
- [x] CLI `cruncher campaign summarize` (aggregate runs).
- [x] Output `campaign_summary.csv` + plots.

### Phase 4 — UX polish
- [x] Add `--category`/`--campaign` filters to `cruncher targets`.
- [x] Add `--campaign` to fetch commands.
- [x] Optional: campaign-level marimo notebook.
- [x] Add `cruncher campaign validate` for preflight selector + cache checks.

### Demo alignment
- [x] Demo workspace includes Category1/2/3 + `demo_categories(_best)` campaign.
- [x] Demo walkthrough covers fetch, selectors, and multi-TF analysis (`docs/demo_campaigns.md`).

## 15) Test plan

- **Campaign expansion**: deterministic output sets; handles overlaps; validates selectors.
- **Generated config**: passes `load_config` and runs parse/sample.
- **Manifest tagging**: run manifests include campaign metadata when used.
- **Pairwise grid plot**: produces output for N>2 TFs (and N=1/2).
- **Joint metrics**: values match expected from synthetic data.
- **Campaign summary**: aggregates multiple runs into correct tables.
- **Notebook**: detects and renders new tables/plots; strict mode still validates.

## 16) Outcomes if implemented

If fully implemented, Cruncher will:
- Support **category-driven campaigns** without changing core optimization.
- Enable scalable **N>2 regulator** workflows with clear projections and joint metrics.
- Provide **campaign-level summaries** for landscape exploration.
- Keep the system **reproducible, decoupled, and assertive** with deterministic manifests and offline analysis.
- Improve UX for TF discovery and selection while preserving existing CLI contracts.

---

## Appendix: File references

- Config validation: `src/config/schema_v2.py:AnalysisConfig._check_tf_pair`
- Analysis layout: `src/utils/analysis_layout.py`
- Analysis workflow: `src/workflows/analyze_workflow.py:run_analyze`
- Plot registry: `src/workflows/analyze/plot_registry.py:PLOT_SPECS`
- Summary plots: `src/workflows/analyze/plots/summary.py:*`
- Notebook: `src/cli/commands/notebook.py`, `src/services/notebook_service.py`
- Run index: `src/services/run_service.py`
