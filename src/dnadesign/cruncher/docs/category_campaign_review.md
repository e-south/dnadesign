# Cruncher category/campaign review (ergonomics + extensibility)

Date: 2026-01-08

## Executive summary

What already works:
- Multi-regulator (N>2) sampling is already supported end-to-end in **parse/sample/analyze/report** as long as `regulator_sets` contains the desired TF list. Sampling/scoring treat TFs as a list and produce per-TF scores for all TFs, not just pairs. This is driven by `regulator_sets` in config and flows through run manifests and artifacts. (refs: `src/config/schema_v2.py:CruncherConfig.regulator_sets`, `src/workflows/sample_workflow.py:_run_sample_for_set`, `src/core/evaluator.py:SequenceEvaluator`, `src/workflows/analyze_workflow.py:run_analyze`, `src/workflows/analyze/plots/summary.py:plot_score_hist`, `src/workflows/report_workflow.py:run_report`)
- Reproducibility boundaries are solid: lockfiles are mandatory for parse/sample and run manifests record resolved motifs and hashes. (refs: `src/store/lockfile.py:validate_lockfile`, `src/workflows/parse_workflow.py:_lockmap_for`, `src/workflows/sample_workflow.py:_lockmap_for`, `src/utils/manifest.py:build_run_manifest`)

What does not work well for the new category/campaign use case:
- There is **no category/campaign abstraction**; users must manually enumerate `regulator_sets`, and the CLI has no helper to expand categories into combinations or to run multi-regulator campaigns at scale. (refs: `src/config/schema_v2.py:CruncherConfig.regulator_sets`, `docs/config.md:Root settings`, `src/cli/commands/lock.py:lock`)
- Analysis/reporting is **mostly per-run** and **pairwise-centric** for many plots. `analysis.tf_pair` is required for pairwise plots and must be exactly two TFs; there is no cross-run “landscape” summary. (refs: `src/config/schema_v2.py:AnalysisConfig._check_tf_pair`, `src/workflows/analyze_workflow.py:_resolve_tf_pair`, `src/workflows/analyze/plot_registry.py:PLOT_SPECS`, `src/workflows/analyze/plots/scatter.py:plot_scatter`)

Recommendation:
- Add a **campaign expansion layer** that materializes category-based combinations into explicit `regulator_sets`, plus a lightweight **campaign summary** command that aggregates run artifacts across many runs. Keep core compute, lockfiles, and run artifacts unchanged.

---

## Current-state map (lifecycle + config → behavior)

### Config inputs and run expansion
- `cruncher.regulator_sets` is the only mechanism to define TF groups; each entry becomes a separate parse/sample run. (refs: `src/config/schema_v2.py:CruncherConfig.regulator_sets`, `docs/config.md:Root settings`, `src/workflows/parse_workflow.py:run_parse`, `src/workflows/sample_workflow.py:run_sample`)
- `regulator_sets` are iterated as lists; no structural constraints beyond non-empty groups. This already supports N>2 TFs. (refs: `src/workflows/parse_workflow.py:run_parse`, `src/workflows/sample_workflow.py:run_sample`, `src/utils/labels.py:regulator_sets`)

### Lifecycle commands
- **fetch** is explicit and requires individual `--tf`/`--motif-id` values; there is no “fetch all configured targets” or category-aware fetch. (refs: `src/cli/commands/fetch.py:motifs`, `src/cli/commands/fetch.py:sites`, `docs/cli.md:cruncher fetch motifs`)
- **lock** resolves *all TF names* appearing in `regulator_sets` and writes a single lockfile per config. (refs: `src/cli/commands/lock.py:lock`, `src/services/lock_service.py:resolve_lock`)
- **parse/sample** run once per `regulator_set` and record the active TF list in the run manifest. (refs: `src/workflows/parse_workflow.py:run_parse`, `src/workflows/sample_workflow.py:_run_sample_for_set`, `src/utils/manifest.py:build_run_manifest`)
- **analyze/report** operate on a *single sample run* at a time. (refs: `src/cli/commands/analyze.py:analyze`, `src/workflows/analyze_workflow.py:run_analyze`, `src/workflows/report_workflow.py:run_report`)

### Artifacts produced (multi-TF aware)
- `sequences.parquet` includes `score_<tf>` columns for each TF in the active set. (ref: `src/workflows/sample_workflow.py:_run_sample_for_set`)
- `elites.parquet` and `elites.json` include per-TF scores for all TFs in the active set. (ref: `src/workflows/sample_workflow.py:_run_sample_for_set`)
- `config_used.yaml` records `pwms_info` for each TF (matrix + consensus). (ref: `src/workflows/sample_workflow.py:_save_config`)
- `run_manifest.json` records each TF’s motif metadata (length, site counts, dataset IDs, tags). (ref: `src/utils/manifest.py:build_run_manifest`)

---

## A) Current capability (with file-level citations)

### Does Cruncher support regulator_sets with N>2 TFs end-to-end?

**Yes, for parse/sample/analyze/report**, as long as the config explicitly provides the set. Evidence:
- `regulator_sets` is a list of TF lists without pairwise restriction. (ref: `src/config/schema_v2.py:CruncherConfig.regulator_sets`)
- `parse` and `sample` iterate through each set, de-duplicate TFs, and operate on arbitrary-length `tfs`. (refs: `src/workflows/parse_workflow.py:run_parse`, `src/workflows/sample_workflow.py:run_sample`, `src/workflows/sample_workflow.py:_run_sample_for_set`)
- `SequenceEvaluator` combines scores across all TFs (default min or sum depending on scale), not pairwise. (ref: `src/core/evaluator.py:SequenceEvaluator`)
- Non-pairwise analysis plots (score hist/box, correlation heatmap, parallel coords) operate on all TF score columns. (refs: `src/workflows/analyze/plots/summary.py:plot_score_hist`, `src/workflows/analyze/plots/summary.py:plot_correlation_heatmap`, `src/workflows/analyze/plots/summary.py:plot_parallel_coords`)
- Reports list TF names from run manifest; they are not limited to two. (ref: `src/workflows/report_workflow.py:run_report`)

### Where “pairwise only” is assumed

Explicit pairwise constraints:
- `analysis.tf_pair` must have exactly two TFs. (ref: `src/config/schema_v2.py:AnalysisConfig._check_tf_pair`)
- Pairwise plots depend on `tf_pair`, enforced before plot generation. (refs: `src/workflows/analyze_workflow.py:_resolve_tf_pair`, `src/workflows/analyze_workflow.py:run_analyze`, `src/workflows/analyze/plot_registry.py:PLOT_SPECS`)
- Scatter and pairwise diagnostics only consume a TF pair. (refs: `src/workflows/analyze/plots/scatter.py:plot_scatter`, `src/workflows/analyze/plots/diagnostics.py:make_pair_idata`)

### Artifacts and summaries for multi-TF runs

- **Run manifest** includes per-TF motif metadata (matrix length, site counts, dataset IDs, tags). (ref: `src/utils/manifest.py:build_run_manifest`)
- **Sequences**: `sequences.parquet` includes per-TF scores for *all* TFs in the active set. (ref: `src/workflows/sample_workflow.py:_run_sample_for_set`)
- **Elites**: `elites.parquet` and `elites.json` include per-TF scores for all TFs. (ref: `src/workflows/sample_workflow.py:_run_sample_for_set`)
- **Analysis summary** records `tf_names` list for the run. (ref: `src/workflows/analyze_workflow.py:run_analyze`)
- **Report** shows TF list and run stats; no multi-run aggregation. (ref: `src/workflows/report_workflow.py:run_report`)

---

## B) Ergonomics gaps for category/campaign workflows

### What’s awkward today

- **No category concept**: Users must explicitly enumerate every combination in `regulator_sets`. This is manual, error-prone, and creates huge configs for pair/triple sweeps. (refs: `src/config/schema_v2.py:CruncherConfig.regulator_sets`, `docs/config.md:Root settings`)
- **No helper for combinatorics**: There is no CLI or workflow to generate combinations (pairs/triples) within or across categories. (refs: `src/cli/commands/lock.py:lock`, `src/cli/commands/targets.py:list_config_targets`)
- **Manual fetch**: Fetching is manual per TF (`--tf`), so category-based fetch requires manual scripting. (refs: `src/cli/commands/fetch.py:motifs`, `src/cli/commands/fetch.py:sites`)

### What breaks down in discovery/run management

- **Run explosion without grouping**: Runs are per regulator set, but no campaign grouping or tagging exists in the run index. (refs: `src/services/run_service.py:RunInfo`, `src/utils/manifest.py:build_run_manifest`)
- **Analysis is per-run**: `analysis.runs` lists explicit run names; no selection by tag/category/campaign. (refs: `src/config/schema_v2.py:AnalysisConfig.runs`, `src/workflows/analyze_workflow.py:run_analyze`)

### What’s missing for cross-run comparison

- **No aggregated summary** across runs (e.g., best joint scores per combination, elite counts by set). The existing `report` and `analysis` are per-run only. (refs: `src/workflows/report_workflow.py:run_report`, `src/workflows/analyze_workflow.py:run_analyze`)
- **No landscape view** (heatmaps, tables, or ranked lists) to compare many combinations. (refs: `src/workflows/analyze/plot_registry.py:PLOT_SPECS`, `src/workflows/analyze/plots/summary.py`)

---

## C) Quality metrics (existing signals + recommended location)

### Signals already computed or easily derived

**From catalog / ingestion (source-agnostic):**
- Matrix length, site counts, site length statistics, dataset IDs, and tags are in catalog entries. (refs: `src/store/catalog_index.py:CatalogEntry`, `src/services/target_service.py:TargetStats`, `src/services/target_service.py:target_stats`)
- Provenance tags (e.g., dataset method/source) are recorded in catalog entries. (ref: `src/store/catalog_index.py:CatalogEntry`)

**From PWM objects / parse/sample:**
- Information content (PWM entropy) is computed via `PWM.information_bits()`. (ref: `src/core/pwm.py:PWM.information_bits`)
- `config_used.yaml` stores PWM matrices and consensus sequences for each TF. (ref: `src/workflows/sample_workflow.py:_save_config`)

**From run artifacts:**
- Per-TF score distributions exist in `sequences.parquet` and can be summarized (mean/median/std/min/max). (ref: `src/workflows/analyze/plots/summary.py:write_score_summary`)
- `run_manifest.json` embeds the catalog metadata for each TF (site counts, matrix length, tags). (ref: `src/utils/manifest.py:build_run_manifest`)

### Where “quality scoring” should live

**Recommended separation (to preserve reproducibility and decoupling):**
- **Catalog-level quality signals** (source-agnostic, stable): compute once at ingest/refresh and store in catalog entries or derived catalog “stats”. This keeps them independent of run-specific settings and makes them reusable across projects. (refs: `src/store/catalog_index.py:CatalogEntry`, `src/services/target_service.py:target_stats`)
- **Run-level quality summaries** (run-specific): compute in analysis (offline) using `sequences.parquet` and `elites.parquet`, then store in `analysis/summary.json` or new `analysis/tables/*.csv`. (refs: `src/workflows/analyze_workflow.py:run_analyze`, `src/workflows/analyze/plots/summary.py:write_score_summary`)

This keeps network access isolated to fetch while making scoring deterministic and reproducible via lockfiles and run manifests.

---

## D) Best-practice design proposal

### Option 1 — Config-driven expansion (minimal UX changes, no new commands)

**Idea:** Add optional `regulator_categories` and `campaigns` in config; expand to `regulator_sets` inside config loading or workflows before lock/parse/sample.

Pros:
- No new CLI required; users keep calling existing commands.
- Keeps workflows intact once expansion is done.

Cons:
- Adds complexity into config parsing and schema validation.
- Harder to inspect or export the expanded sets; users may want a concrete list for review.

### Option 2 — New CLI helper to materialize regulator_sets (recommended)

**Idea:** Introduce `cruncher campaign generate` (or `cruncher expand`) that reads categories + campaign rules, writes a derived config with explicit `regulator_sets`, and records a campaign manifest. Existing lifecycle commands remain unchanged.

Pros:
- Minimal changes to core workflows.
- Clear reproducibility: generated config + manifest can be versioned.
- Keeps network decoupled from compute (generation is offline).

Cons:
- Adds an extra step (generate → lock → parse/sample).

### Option 3 — First-class campaign concept with aggregated analysis

**Idea:** Add a campaign object in config + campaign runs, and implement new `cruncher campaign run/analyze/report` commands that group runs and produce aggregate reports.

Pros:
- Best user experience for large sweeps.
- Enables integrated “landscape” reports and consistent run grouping.

Cons:
- Larger surface-area change; more schema and workflow extensions.

### Primary recommendation: Option 2 + lightweight campaign summary

Implement a CLI helper to generate explicit `regulator_sets` **and** add a small, offline “campaign summary” step that aggregates metrics across many runs. This keeps compute decoupled and avoids changes to core sampling logic.

---

## Proposed config additions (additive)

### YAML shape (additive, optional)

```yaml
cruncher:
  out_dir: runs/

  regulator_categories:
    Category1: [CpxR, BaeR]
    Category2: [LexA, RcdA, Lrp, Fur]
    Category3: [Fnr, Fur, AcrR, SoxR, SoxS, Lrp]

  campaigns:
    - name: regulators_v1
      categories:
        - Category1
        - Category2
        - Category3
      within_category:
        sizes: [2, 3]
      across_categories:
        sizes: [2, 3]
        max_per_category: 2
      allow_overlap: true
      dedupe_sets: true
      tags:
        organism: ecoli
        purpose: "multi-regulator sweep"

  # (Optional) explicit regulator_sets can still be provided
  regulator_sets: []
```

Notes:
- `regulator_sets` remains the runtime contract. Campaigns only **generate** it.
- Overlapping regulators (Fur, Lrp) are allowed; `allow_overlap: true` controls whether a TF can appear multiple times across categories.

---

## Proposed CLI UX (minimal additions)

### 1) Generate explicit regulator_sets

```
cruncher campaign generate --campaign regulators_v1 --out config.generated.yaml
```

Behavior:
- Reads `regulator_categories` + `campaigns` and writes an expanded config with explicit `regulator_sets`.
- Writes a campaign manifest under `.cruncher/campaigns/<campaign_id>.json` (or alongside the generated config).
- No network access.

### 2) Optional: fetch all campaign targets

```
cruncher fetch motifs --targets --campaign regulators_v1 <config>
cruncher fetch sites  --targets --campaign regulators_v1 <config>
```

Behavior:
- Resolves the TF list from the campaign or expanded config.
- Calls existing fetch logic per TF.

### 3) Aggregate “landscape” summary

```
cruncher campaign summarize --campaign regulators_v1 --runs runs/*
```

Behavior:
- Consumes run artifacts (`run_manifest.json`, `sequences.parquet`, `elites.parquet`) and produces:
  - `campaign_summary.csv` (per-run metrics)
  - `campaign_best.csv` (top combinations)
  - Optional plots (heatmap / top-K bar chart)
- No network access.

---

## Implementation locations (minimal touch points)

### Config/schema
- Add optional `regulator_categories` and `campaigns` to config schema. (new fields in `src/config/schema_v2.py:CruncherConfig`)

### Campaign expansion
- New service module: `src/services/campaign_service.py` (expand categories → list of TF sets, dedupe, validate).
- CLI command: `src/cli/commands/campaign.py` (generate, summarize).

### Lockfiles and manifests
- **Unchanged lockfile format**: `cruncher lock` already resolves the union of TFs in `regulator_sets`. (ref: `src/cli/commands/lock.py:lock`)
- Add optional `campaign` metadata into run manifest extras so runs can be grouped later. (ref: `src/utils/manifest.py:build_run_manifest`)

### Analysis summaries
- Add a new workflow `src/workflows/campaign_summary.py` that reads existing run artifacts and writes aggregate tables/plots.
- Keep `analyze` and `report` unchanged for per-run diagnostics. (refs: `src/workflows/analyze_workflow.py:run_analyze`, `src/workflows/report_workflow.py:run_report`)

---

## Proposed analysis/report outputs for “surveying the landscape”

### Output layout (new)

```
<out_dir>/campaigns/<campaign_id>/
  campaign_manifest.json
  campaign_summary.csv
  campaign_best.csv
  plots/
    best_normsum_bar.png
    tf_coverage_heatmap.png
```

### Metrics to include (all derivable offline)

Per-run rows in `campaign_summary.csv`:
- run_name, set_index, tf_list
- n_tfs, sequence_length
- n_sequences, n_elites
- best_norm_sum (from elites)
- mean_score_<tf> / median_score_<tf> (from sequences.parquet)
- pwm_info_bits_<tf> (from PWM matrices in config_used.yaml)
- site_count_<tf> (from run manifest motifs)

Per-run summary can be computed without touching ingestion or network.

---

## Explicit “unchanged” components

- Core PWM scoring and MCMC optimization (no changes). (refs: `src/core/evaluator.py:SequenceEvaluator`, `src/core/scoring.py:Scorer`)
- Lockfile semantics and verification. (refs: `src/store/lockfile.py:validate_lockfile`, `src/store/lockfile.py:verify_lockfile_hashes`)
- Existing parse/sample/analyze/report stages and artifacts.

---

## Implementation plan (phased, minimal)

**Phase 1 — Expansion + metadata (no core changes)**
1. Add optional config schema fields for `regulator_categories` and `campaigns`.
2. Implement `campaign_service.expand_campaign(...)` to produce explicit TF sets.
3. Add CLI `cruncher campaign generate` to write a derived config + manifest.
4. Add run manifest extras for `campaign_id`/`campaign_entry` when using generated config.

**Phase 2 — Landscape summaries (offline)**
5. Implement `cruncher campaign summarize` to aggregate run artifacts into CSV/plots.
6. Add doc updates + examples.

**Phase 3 — UX polish**
7. Optional fetch helper: `cruncher fetch motifs/sites --targets --campaign <name>`.
8. Optional “runs list --campaign <name>” filter (reading campaign metadata from run_index).

---

## Test plan

Add/modify tests to cover:
- **Campaign expansion**: given categories + rules, expansion produces deterministic regulator_sets; overlaps handled. (new tests in `tests/test_campaign_expand.py`)
- **Generated config**: output YAML contains explicit `regulator_sets` and is still accepted by `load_config`. (tests around `src/config/schema_v2.py:CruncherConfig`)
- **Manifest tagging**: sample runs store campaign metadata in `run_manifest.json` when generated config is used. (tests similar to `tests/test_regulator_sets_runs.py`)
- **Campaign summary**: aggregate summary produces expected columns from synthetic run artifacts. (new tests in `tests/test_campaign_summary.py`)

---

## Appendix: Notable current constraints (for awareness)

- Pairwise plots require `analysis.tf_pair` with exactly two TFs. (refs: `src/config/schema_v2.py:AnalysisConfig._check_tf_pair`, `src/workflows/analyze/plot_registry.py:PLOT_SPECS`)
- Fetching is TF-by-TF unless manually scripted; no “all configured targets” helper. (refs: `src/cli/commands/fetch.py:motifs`, `src/cli/commands/fetch.py:sites`)
- Run naming is based on TF slug + set index, which becomes unwieldy for large sets. (ref: `src/utils/labels.py:build_run_name`)
