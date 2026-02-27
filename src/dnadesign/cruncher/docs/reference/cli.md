## Cruncher CLI

**Last updated by:** cruncher-maintainers on 2026-02-23

### Contents
- [Cruncher CLI](#cruncher-cli)
- [Workspace discovery and config resolution](#workspace-discovery-and-config-resolution)
- [Quick command map](#quick-command-map)
- [Core lifecycle commands](#core-lifecycle-commands)
- [Study workflows](#study-workflows)
- [Portfolio workflows](#portfolio-workflows)
- [Discovery and inspection](#discovery-and-inspection)
- [Global options](#global-options)

This reference summarizes the Cruncher CLI surface, grouped by lifecycle stage and workflow.

> **Intent:** Cruncher is an optimization engine for **fixed-length** multi-TF PWM sequence design that returns a **diverse elite set** - not posterior inference.
>
> **When to use:** design under tight length constraints; explore motif compatibility tradeoffs; generate a small candidate set for assays; run workspace-scoped studies and summarize aggregate outcomes.

#### Workspace discovery and config resolution

Cruncher resolves config from `--config/-c` or `--workspace/-w`, then `<workspace>/configs/config.yaml` from the current directory/parents (or `config.yaml` when CWD is already `<workspace>/configs`), then known workspace roots. If multiple workspaces are found, **cruncher** prompts for a selection (interactive shells only).

See available workspaces with:

```
cruncher workspaces list
```

List workspace-scoped Study specs and Study runs with:

```
cruncher study list
```

---

#### Quick command map

* **Cache data** → `fetch motifs` / `fetch sites`
* **Inspect cache** → `sources ...` / `catalog ...`
* **Pin TFs** → `lock`
* **Validate motifs** → `parse`
* **Render logos** → `catalog logos`
* **Optimize** → `sample`
* **Analyze** → `analyze`, `notebook`
* **Study sweeps** → `study list|run|summarize|show|clean`
* **Cross-workspace handoff aggregation** → `portfolio run|show`
* **Export sequences** → `export sequences`
* **Run management** → `runs list/show/latest/best/watch/clean`
* **Workspace health + machine runbooks** → `status`, `workspaces run|reset`

---

#### Core lifecycle commands

#### `cruncher fetch motifs`

Caches motif matrices into `<catalog.root>/normalized/motifs/...`.

Inputs:

* optional config path (`--config/-c`), otherwise resolved from workspace/CWD
* at least one of `--tf` or `--motif-id`

Network:

* yes by default; use `--offline` to restrict to cached motifs only

When to use:

* you want `cruncher.catalog.pwm_source: matrix`
* you want to reuse alignment/matrix payloads across runs

Examples:

* `cruncher fetch motifs --tf lexA --tf cpxR <config>`
* `cruncher fetch motifs --motif-id RDBECOLITFC00214 <config>`
* `cruncher fetch motifs --source omalley_ecoli_meme --tf lexA <config>`
* `cruncher fetch motifs --dry-run --tf lexA <config>`

Common options:

* `--tf`, `--motif-id`, `--source`
* `--dry-run`, `--all`, `--offline`, `--update`
* `--summary/--no-summary`, `--paths`

Outputs:

* writes cached motif JSON files and updates `catalog.json`
* prints a summary table by default (or raw paths with `--paths`)

Note:

* `--source` defaults to the first available entry in `catalog.source_preference` (skipping entries that are
  not registered ingest sources); if the list is empty or none are available you must pass `--source` explicitly.

---

#### `cruncher fetch sites`

Caches binding-site instances into `<catalog.root>/normalized/sites/...`.

Inputs:

* optional config path (`--config/-c`), otherwise resolved from workspace/CWD
* at least one of `--tf`, `--motif-id`, or `--hydrate`

Network:

* yes by default; use `--offline` to restrict to cached sites only

When to use:

* you want `cruncher.catalog.pwm_source: sites`
* you want curated or HT site sets cached locally
* you need hydration for coordinate-only peaks

Examples:

* `cruncher fetch sites --tf lexA --tf cpxR <config>`
* `cruncher fetch sites --dry-run --tf lexA <config>`
* `cruncher fetch sites --dataset-id <id> --tf lexA <config>`
* `cruncher fetch sites --genome-fasta genome.fna <config>`

Common options:

* `--tf`, `--motif-id`, `--dataset-id`, `--limit`, `--source`
* `--hydrate` (hydrates missing sequences)
* `--offline`, `--update`
* `--genome-fasta`
* `--summary/--no-summary`, `--paths`

Outputs:

* writes cached site JSONL files and updates `catalog.json`
* prints a summary table by default (or raw paths with `--paths`)

Note:

* `--hydrate` with no `--tf/--motif-id` hydrates all cached site sets by default.
* `--source` defaults to the first available entry in `catalog.source_preference` (skipping entries that are
  not registered ingest sources); if the list is empty or none are available you must pass `--source` explicitly.
* with both curated and HT enabled, `--limit` requires explicit mode (`--dataset-id` or one source class disabled).
* HT mode is strict: if HT discovery/fetch fails or returns zero rows for the selected mode, the command errors.
* if `tfbinding` returns zero rows for a known dataset, switch `ingest.regulondb.ht_binding_mode` to `peaks`.

---

#### `cruncher lock`

Resolves TF names to exact cached artifacts (IDs + hashes) from `workspace.regulator_sets`.
Writes `<workspace>/.cruncher/locks/<config>.lock.json`.

Inputs:

* optional config path (`--config/-c`), otherwise resolved from workspace/CWD
* cached motifs/sites for the configured regulators

Network:

* no (cache-only)

When to use:

* before `parse` and `sample`
* whenever you change anything affecting TF resolution (PWM source, site kinds, dataset selection, etc.)

Example:

* `cruncher lock <config>`

---

#### `cruncher parse`

Validates cached PWMs (matrix- or site-derived) and writes parse-cache artifacts in workspace state.

Inputs:

* CONFIG (explicit or resolved)
* lockfile (from `cruncher lock`)
* optional `--force-overwrite` to replace an existing run directory

Network:

* no (cache-only)

Example:

* `cruncher parse <config>`

Precondition:

* lockfile exists (`cruncher lock <config>`)

Notes:

* Logos are rendered via `cruncher catalog logos`; default logo settings are `--bits-mode information` and `--dpi 150`.
* `cruncher parse` always uses the lockfile to pin exact motif IDs/hashes.
  If you add new motifs (e.g., via `discover motifs`) or change `catalog` preferences,
  re-run `cruncher lock <config>` to refresh what parse will use.
* Parse requires overwrite intent when output already exists; use `--force-overwrite` to replace.
* Parse artifacts live under `<workspace>/.cruncher/parse/inputs/` and are intentionally
  separate from user-facing sample outputs in `workspace.out_dir`.
* Use `cruncher catalog logos` to render PWM logos with provenance subtitles.

---

#### `cruncher sample`

Runs MCMC optimization to design sequences scoring well across TFs.

Inputs:

* CONFIG (explicit or resolved)
* lockfile (from `cruncher lock`)

Network:

* no (cache-only)

Example:

* `cruncher sample <config>`
* `cruncher sample --verbose <config>`
* `cruncher sample --no-progress <config>`
* `cruncher sample --debug <config>`

Precondition:

* lockfile exists (`cruncher lock <config>`)

Notes:

* `sample.output.save_sequences: true` is required for later analysis.
* `sample.output.save_trace: true` enables trace-based diagnostics.
* `--no-progress` disables progress bars and periodic progress logging for quieter non-interactive runs.
* `sample.output.save_trace: false` skips ArviZ trace construction and reduces sample runtime/memory overhead.
* Sampling uses `sample.optimizer.*` (`kind: gibbs_anneal`) with chain count and cooling schedule under one explicit surface.
* Replica exchange is disabled in `gibbs_anneal`; chains are tracked directly in trajectory outputs.
* `--verbose` enables periodic progress logging; `--debug` enables very verbose debug logs.

---

#### `cruncher analyze`

Generates diagnostics and plots for one or more sample runs.

Inputs:

* CONFIG (explicit or resolved)
* runs via `analysis.run_selector`/`analysis.runs` or `--run` (defaults to latest sample run if empty)
* run artifacts: `optimize/tables/sequences.parquet` (required), `optimize/tables/elites.parquet` (required),
  `optimize/tables/elites_hits.parquet` (required), `optimize/tables/random_baseline*.parquet` (required when
  `sample.output.save_random_baseline=true`, default with `sample.output.random_baseline_n=10000`), and
  `optimize/trace.nc` for trace-based plots

Network:

* no (run artifacts only)

Examples:

* `cruncher analyze --latest <config>`
* `cruncher analyze --run <run_name|run_dir> <config>`
* `cruncher analyze --summary <config>`

Preconditions:

* provide runs via `analysis.runs`/`--run` or rely on the default latest run
* selected sample runs must be completed; analyze fails fast when latest run status is still `running`
* run selection preflight happens before plotting/cache initialization, so missing/incomplete runs fail quickly with no Matplotlib/ArviZ cache setup
* trace-dependent plots require `optimize/trace.nc`
* each sample run snapshots the lockfile under `provenance/lockfile.json`; analysis uses that snapshot to avoid mismatch if the workspace lockfile changes later
* if `analysis` is omitted from config, analyze uses schema defaults (including `run_selector=latest`)
* if `analysis.fimo_compare.enabled=true`, MEME Suite `fimo` must be resolvable via `discover.tool_path`, `MEME_BIN`, or `PATH`

Outputs:

* tables: `analysis/tables/table__scores_summary.parquet`, `analysis/tables/table__elites_topk.parquet`,
  `analysis/tables/table__metrics_joint.parquet`, `analysis/tables/table__chain_trajectory_points.parquet`,
  `analysis/tables/table__chain_trajectory_lines.parquet`,
  `analysis/tables/table__diagnostics_summary.json`, `analysis/tables/table__objective_components.json`,
  `analysis/tables/table__elites_mmr_summary.parquet`, `analysis/tables/table__elites_nn_distance.parquet`,
  `analysis/tables/table__elites_mmr_sweep.parquet` (when `analysis.mmr_sweep.enabled=true`)
* plots: `plots/elite_score_space_context.<plot_format>`,
  `plots/chain_trajectory_sweep.<plot_format>`,
  `plots/elites_nn_distance.<plot_format>`, `plots/elites_showcase.<plot_format>`,
  `plots/health_panel.<plot_format>` (trace only)
* reports: `analysis/reports/report.json`, `analysis/reports/report.md`
* summaries: `analysis/reports/summary.json`, `analysis/manifests/manifest.json`, `analysis/manifests/plot_manifest.json`, `analysis/manifests/table_manifest.json`

Note:

* Analyze rewrites the latest analysis outputs each run; set `analysis.archive=true` to keep prior reports.
* Analyze uses `<run_dir>/analysis/state/tmp` as a run-local lock. Stale temp locks from interrupted runs are auto-pruned; active locks still fail fast.
* Use `cruncher analyze --summary` to print the highlights from `report.json`.

---

#### Study workflows

#### `cruncher study list`

Lists Study specs and Study runs discovered under known workspace roots.

Inputs:

* optional `--workspace <name|index|path>` to scope listing to one workspace

Network:

* no

Examples:

* `cruncher study list`
* `cruncher study list --workspace demo_pairwise`

Notes:

* Studies are workspace-scoped: specs live under `<workspace>/configs/studies/*.study.yaml`.
* Study runs live under the canonical root `<workspace>/outputs/studies/<study_name>/<study_id>/`.
* `--workspace` accepts a discovered workspace name/index or a direct workspace path.
* `study list` fails fast if a discovered spec or run metadata is invalid.

#### `cruncher study run`

Executes a Study spec (`*.study.yaml`) to run sweep factors x replicate seeds, optional replay sweeps, and aggregate plots.

Inputs:

* required `--spec <workspace>/configs/studies/<name>.study.yaml`
* optional `--resume` to continue from an existing manifest
* optional `--force-overwrite` to delete/recreate the deterministic study run dir

Network:

* no (uses local cache and local run artifacts)

Examples:

* `cruncher study run --spec configs/studies/diversity_vs_score.study.yaml`
* `cruncher study run --spec configs/studies/diversity_vs_score.study.yaml --resume`
* `cruncher study run --spec configs/studies/diversity_vs_score.study.yaml --force-overwrite`

Outputs:

* `<workspace>/outputs/studies/<study_name>/<study_id>/study/spec_frozen.yaml`
* `<workspace>/outputs/studies/<study_name>/<study_id>/study/study_manifest.json`
* `<workspace>/outputs/studies/<study_name>/<study_id>/tables/table__trial_metrics.parquet`
* `<workspace>/outputs/plots/study__<study_name>__<study_id>__plot__sequence_length_tradeoff.pdf`
* `<workspace>/outputs/plots/study__<study_name>__<study_id>__plot__mmr_diversity_tradeoff.pdf` (when replay enabled)

Notes:

* Study specs are strict (`extra=forbid`); unknown keys and invalid factor dot-paths fail fast.
* `study.base_config` is required and must exist; no CWD fallback.
* Trial definitions can be explicit (`trials`) or grid-expanded (`trial_grids`); at least one source is required.
* `study.schema_version` is v3 only (`study.schema_version: 3`).
* Trial and grid definitions use `factors`, not `overrides`.
* Studies inherit non-swept behavior from `configs/config.yaml`; only sweep-factor keys are allowed in study specs.
* Every swept factor must include the base-config value in the study domain.
* Trial-grid expansion is bounded (`<=500` combinations per grid and `<=500` total expanded trials).
* Study trials do not register entries in workspace `run_index.json`.
* `study.replays.mmr_sweep.enabled=true` requires persisted sequence artifacts (`sample.output.save_sequences=true`) for every trial after profile factors and requires replay diversity values to include the base-config diversity.
* Preflight validates lockfile, target readiness, and parse-readiness before any trial executes.
* When any trial fails, `study run` records errors and skips automatic summary generation; run `study summarize --allow-partial` to summarize successful subsets.

#### `cruncher study summarize`

Recomputes aggregate Study tables/plots from an existing study run directory.

Inputs:

* required `--run <study_run_dir>`
* optional `--allow-partial` to include only successful trials when some runs/artifacts are missing

Behavior:

* with `--allow-partial`, aggregate tables include `n_missing_*` annotations (`n_missing_total`, `n_missing_non_success`, `n_missing_run_dirs`, `n_missing_metric_artifacts`, `n_missing_mmr_tables`).
* if partial data was required and the frozen spec uses `exit_code_policy=nonzero_if_any_error`, command exits non-zero after writing refreshed outputs.

Example:

* `cruncher study summarize --run outputs/studies/diversity_vs_score/<study_id>`

#### `cruncher study show`

Prints Study status, trial counts, and key table/plot paths.

Inputs:

* required `--run <study_run_dir>`

Example:

* `cruncher study show --run outputs/studies/diversity_vs_score/<study_id>`

#### `cruncher study clean`

Deletes Study output artifact directories under `outputs/studies/...` for one workspace/study target.

Inputs:

* required `--workspace <name|index|path>`
* required `--study <study_name>`
* exactly one of:
  * `--id <study_id>` for one run
  * `--all` for all runs for that study in that workspace
* optional `--confirm` to execute deletion (without `--confirm`, command is dry-run only)

Behavior:

* cleans output artifacts only; never modifies `*.study.yaml`
* fail-fast contract for invalid workspace selector, missing study spec, missing run, or invalid flag combinations

Examples:

* `cruncher study clean --workspace demo_pairwise --study diversity_vs_score --id <study_id>`
* `cruncher study clean --workspace demo_pairwise --study diversity_vs_score --all --confirm`

---

#### Portfolio workflows

#### `cruncher portfolio run`

Aggregates selected completed runs across multiple workspaces into a deterministic handoff package.

Inputs:

* required `--spec <portfolio_workspace>/configs/<name>.portfolio.yaml`
* optional `--force-overwrite` to delete/recreate the deterministic portfolio run dir
* optional `--prepare-ready {prompt|skip|rerun}` for `prepare_then_aggregate` when some sources are already ready

Network:

* no (run artifacts only)

Examples:

* `cruncher portfolio run --spec configs/master_all_workspaces.portfolio.yaml`
* `cruncher portfolio run --spec configs/master_all_workspaces.portfolio.yaml --force-overwrite`
* `cruncher portfolio run --spec configs/master_all_workspaces.portfolio.yaml --prepare-ready skip`

Source preconditions (per source entry in spec):

* `analysis/reports/summary.json`
* `export/export_manifest.json` (with valid `files.elites` path)

Outputs:

* `<portfolio_workspace>/outputs/portfolios/<portfolio_name>/<portfolio_id>/meta/manifest.json`
* `<portfolio_workspace>/outputs/portfolios/<portfolio_name>/<portfolio_id>/meta/status.json`
* `<portfolio_workspace>/outputs/portfolios/<portfolio_name>/<portfolio_id>/meta/logs/prepare__<source_id>.log` (when `execution.mode=prepare_then_aggregate`)
* `<portfolio_workspace>/outputs/portfolios/<portfolio_name>/<portfolio_id>/tables/table__handoff_windows_long.<csv|parquet>`
* `<portfolio_workspace>/outputs/portfolios/<portfolio_name>/<portfolio_id>/tables/table__handoff_elites_summary.<csv|parquet>`
* `<portfolio_workspace>/outputs/portfolios/<portfolio_name>/<portfolio_id>/tables/table__source_summary.<csv|parquet>`
* `<portfolio_workspace>/outputs/portfolios/<portfolio_name>/<portfolio_id>/tables/table__study_summary.<csv|parquet>` (when source `study_spec` is declared)
* `<portfolio_workspace>/outputs/portfolios/<portfolio_name>/<portfolio_id>/tables/table__handoff_sequence_length.<csv|parquet>` (when `studies.sequence_length_table.enabled: true`)
* `<portfolio_workspace>/outputs/portfolios/<portfolio_name>/<portfolio_id>/plots/plot__source_tradeoff_score_vs_diversity.pdf` (when source diversity metrics are available)
* `<portfolio_workspace>/outputs/portfolios/<portfolio_name>/<portfolio_id>/plots/plot__elite_showcase_cross_workspace.<pdf|png>` (when `plots.elite_showcase.enabled: true`)

Default portfolio table output is parquet without CSV mirrors (`artifacts.table_format=parquet`, `artifacts.write_csv=false`).

Notes:

* Portfolio specs are strict (`extra=forbid`); unknown keys and invalid paths fail fast.
* Sources are explicit only: no latest-run fallback, no workspace auto-selection fallback.
* Source selection uses source run manifest `top_k` and export manifest `files.elites`; there is no portfolio-level top-k setting.
* Source run manifest `top_k` must match export elites row count for each source.
* Source run manifest stage must be `sample`.
* `run_dir` must resolve inside its declared `workspace`.
* For single-set workspaces, use `run_dir: outputs`; for multi-set workspaces use the specific set path (for example `outputs/set2_lexA-cpxR`).
* Portfolio schema is v3 only (`portfolio.schema_version: 3`).
* `execution.mode`:
  * `aggregate_only`: aggregate current source runs only
  * `prepare_then_aggregate`: execute each source `prepare.runbook` + `prepare.step_ids` before aggregation
* `execution.max_parallel_sources` controls source preparation concurrency in `prepare_then_aggregate` (default `4`, must be `>= 1`).
* In `prepare_then_aggregate`, every source must provide a runbook path inside its source workspace and a non-empty step list.
* Optional source `study_spec` adds deterministic study summary rows into `table__study_summary`.
* If `study_spec` is declared, the deterministic study run and `table__trial_metrics_agg.parquet` must exist (or be produced by prepare steps).
* Optional `studies.ensure_specs` enforces workspace-scoped study specs per source and auto-runs/resumes those study runs when missing/incomplete.
* Optional `studies.sequence_length_table` is global at portfolio scope and selects the first `top_n_lengths` shortest `sequence_length` rows per source.
* `plots.elite_showcase` is enabled by default and renders a cross-workspace showcase using all processed source elites.
* Set `plots.elite_showcase.top_n_per_source` to a positive integer when you want to cap elites per source.
* `plots.elite_showcase.source_selectors` supports per-source multi-elite selection, with exactly one of `elite_ids` or `elite_ranks` per source selector.
* In `aggregate_only`, Cruncher preflights all listed sources and reports every missing/invalid source artifact with actionable nudges.
* In `prepare_then_aggregate`, `--prepare-ready` controls whether already-ready sources are reprocessed or skipped.
* If source preparation fails, Cruncher reports source id, runbook path, configured `step_ids`, preflight issues, and explicit `workspaces run` nudge commands.
* Portfolio nudges use `--workspace <source_workspace_path>` (resolved path), not workspace-name lookup, so they remain runnable for external workspaces.
* When preflight shows missing foundational run artifacts (for example missing run manifest/elites), the failure message includes a full-runbook nudge in addition to configured-step nudges.
* `--spec` must point to a `.portfolio.yaml` file path; passing a directory path fails fast with an explicit nudge.

#### `cruncher portfolio show`

Prints portfolio status plus table/plot paths for one portfolio run directory.

Inputs:

* required `--run <portfolio_run_dir>`

Example:

* `cruncher portfolio show --run outputs/portfolios/master_all_workspaces/<portfolio_id>`

---

#### `cruncher export sequences`

Exports sequence-centric run tables for downstream wrappers and operators.

Inputs:

* CONFIG (explicit or resolved)
* exactly one run selector: `--run <run_name|run_dir>` (repeatable) or `--latest`
* sample run artifacts: `optimize/tables/elites.parquet`, `optimize/tables/elites_hits.parquet`, `meta/config_used.yaml`

Network:

* no (run artifacts only)

Examples:

* `cruncher export sequences --latest <config>`
* `cruncher export sequences --run sample/run_001 <config>`
* `cruncher export sequences --latest --table-format csv <config>`

Outputs (under each run):

* `export/table__elites.csv`
* `export/table__consensus_sites.<parquet|csv>`
* `export/export_manifest.json`

Notes:

* Fail-fast contract: duplicate `(elite_id, tf)` rows, out-of-bounds windows, non-numeric scores, or inconsistent motif widths terminate export with an explicit error.
* Export appends artifact entries to `meta/run_manifest.json` using stage `export`.
* Default table format is CSV unless `--table-format parquet` is set.

---

#### `cruncher notebook`

Generates a marimo notebook for interactive exploration.

Inputs:

* run directory (`<run_dir>`) and optional `--analysis-id` or `--latest`

Network:

* no (local artifacts only; marimo is a local dependency)

Example:

* `cruncher notebook <path/to/sample_run> --latest`

Notes:

* requires `marimo` to be installed (for example: `uv sync --locked`)
* useful when you want interactive slicing/filtering beyond static plots
* strict artifact contract: requires `analysis/reports/summary.json`, `analysis/manifests/plot_manifest.json`, and `analysis/manifests/table_manifest.json` to exist and parse, `analysis/reports/summary.json` must include a non-empty `tf_names` list, and `analysis/manifests/table_manifest.json` must provide `scores_summary`, `metrics_joint`, and `elites_topk` entries with existing files
* plot output status is refreshed from disk so missing files are shown accurately
* the Refresh button re-scans analysis entries and updates plot/table status without restarting marimo
* the notebook infers `run_dir` from its location; keep it under `<run_dir>/` or regenerate it
* plots are loaded from `analysis/manifests/plot_manifest.json`; the curated keys are `elite_score_space_context`, `chain_trajectory_sweep`, `elites_nn_distance`, `elites_showcase`, plus optional `health_panel` and `optimizer_vs_fimo` entries when generated
* the notebook includes:
  * Overview tab with run metadata and explicit warnings for missing/invalid analysis artifacts
  * Tables tab with a Top-K slider and per-table previews from `analysis/manifests/table_manifest.json`
  * Plots tab with inline previews and generated/skipped status from `analysis/manifests/plot_manifest.json`

---

#### Discovery and inspection

#### `cruncher workspaces`

List discoverable workspaces and their config paths.

Inputs:

* optional `--root <workspace_parent_dir>` for explicit workspace discovery scope

Network:

* no

Example:

* `cruncher workspaces list`
* `cruncher workspaces list --root src/dnadesign/cruncher/workspaces`
* `cruncher workspaces run --runbook configs/runbook.yaml`
* `cruncher workspaces run --workspace demo_pairwise --step analyze_summary --step export_sequences_latest`
* `cruncher workspaces reset --root .`
* `cruncher workspaces reset --root . --confirm`
* `cruncher workspaces reset --root src/dnadesign/cruncher/workspaces --all-workspaces --confirm`

Notes:

* `workspaces list` includes Study inventory columns: `Study Specs` and `Study Runs`, and reports workspace kind (`config+runbook` or `runbook-only`).
* `workspaces run` executes typed runbook steps from `configs/runbook.yaml` in fail-fast order.
* runbook steps are strict CLI-args only (`run: [<cruncher-subcommand>, ...]`); arbitrary shell is not supported.
* `workspaces run --step ...` filters to explicit step ids while preserving runbook order.
* when `--workspace` and a relative `--runbook` are both provided, `--runbook` resolves from the selected workspace root and must stay inside that workspace.
* `workspaces reset` is a confirm-gated workspace reset surface that preserves `inputs/` and `configs/` while removing generated state (`outputs/`, `.cruncher/`, transient cache files).
* `workspaces reset --all-workspaces` treats `--root` as a parent directory and applies the same reset contract to every discoverable child workspace.

---

#### `cruncher catalog`

Inspect cached motifs and site sets.

Use `catalog pwms` to compute PWMs from cached matrices or binding sites and
survey their lengths/bit scores (without sampling-time motif-width trimming), and `catalog logos` to render PNG logos for the
same selection criteria.

Inputs:

* CONFIG (explicit or resolved)

Network:

* no (catalog only)

Subcommands:

* `catalog list` — list cached motifs and site sets
* `catalog search` — search by TF name or motif ID
* `catalog resolve` — resolve a TF name to cached candidates
* `catalog show` — show metadata for a cached `<source>:<motif_id>`
* `catalog pwms` — summarize or export resolved PWMs (matrix or site-derived)
* `catalog export-densegen` — export DenseGen motif artifacts (one JSON per motif)
* `catalog export-sites` — export cached binding sites as CSV/Parquet for DenseGen
* `catalog logos` — render PWM logos for selected TFs or motif refs

Note: `catalog logos` is idempotent for identical inputs. If matching logos already exist
under `outputs/plots/`, it reports the existing path instead of writing new files.

Examples:

* `cruncher catalog list <config>`
* `cruncher catalog search <config> lexA --fuzzy`
* `cruncher catalog show <config> regulondb:RDBECOLITFC00214`
* `cruncher catalog pwms <config>`
* `cruncher catalog pwms --set 1 <config>`
* `cruncher catalog export-sites --set 1 --out densegen/sites.csv <config>`
* `cruncher catalog export-sites --set 1 --densegen-workspace demo_tfbs_baseline <config>`
* `cruncher catalog export-densegen --set 1 --out densegen/pwms <config>`
* `cruncher catalog export-densegen --set 1 --densegen-workspace demo_sampling_baseline <config>`
* `cruncher catalog logos --set 1 <config>`

`catalog export-densegen` and `catalog export-sites` accept `--densegen-workspace` (packaged DenseGen
workspace name, explicit workspace path, or name under `DNADESIGN_DENSEGEN_WORKSPACES_ROOT`).
When provided, outputs default to the workspace `inputs/` locations and must stay within that directory.
`catalog export-densegen` removes existing artifact JSONs for the selected TFs by default; use
`--no-clean` to keep prior artifacts.

---

#### `cruncher discover`

Discover motifs from cached binding sites using MEME Suite (STREME or MEME).

Inputs:

* CONFIG (explicit or resolved)

Network:

* no (local; requires MEME Suite CLI tools via PATH or tool_path/MEME_BIN)

Subcommands:

* `discover motifs` — run STREME/MEME per TF and ingest discovered motifs into the catalog
* `discover check` — validate that MEME Suite tools are available and report versions

Examples:

* `cruncher discover motifs --set 1 <config>`
* `cruncher discover motifs --tf lexA --tf cpxR --tool streme <config>`
* `cruncher discover motifs --tf lexA --tf cpxR --tool meme <config>`
* `cruncher discover motifs --tf lexA --tf cpxR --tool meme --meme-mod oops <config>`
* `cruncher discover motifs --tf lexA --tf cpxR --tool meme --meme-mod oops --meme-prior addone <config>`
* `cruncher discover motifs --tf lexA --tf cpxR --tool streme --source-id meme_suite_streme <config>`
* `cruncher discover motifs --tf lexA --tf cpxR --tool meme --meme-mod oops --meme-prior addone --source-id meme_suite_meme <config>`
* `cruncher discover motifs --tf lexA --tool streme --replace-existing <config>`
* `cruncher discover motifs --tool-path /opt/meme/bin --tool streme <config>`
* `cruncher discover check <config>`

Notes:
* `tool=auto` selects STREME when there are enough sequences; use `--tool meme` if STREME is not installed.
* Discovery reads cached binding sites (run `cruncher fetch sites` first).
  Discovery always uses cached sites regardless of `catalog.pwm_source`.
* By default discovery uses raw cached site sequences. Use `--window-sites` (or
  `discover.window_sites=true`) to pre-window with `catalog.site_window_lengths`
  before running MEME/STREME.
  If enabled without window lengths for a TF, discovery exits with a helpful error.
* If `--minw/--maxw` are omitted (and unset in config), Cruncher passes no width flags and
  MEME/STREME uses its own defaults.
* `discover motifs` output `Tool width` is the discovery-time motif length from MEME/STREME;
  `Width bounds` reports `minw/maxw` used for discovery (`tool_default` means unset).
  Sampling-time constraints (`sample.motif_width`) are applied later during `sample`.
* Use `cruncher targets stats` to set `--minw/--maxw` from site-length ranges.
* If you plan to run both MEME and STREME, set distinct `discover.source_id` values between runs to avoid lock ambiguity.
  You can also pass `--source-id` per run to avoid editing config.
* By default discovery replaces previous discovered motifs for the same TF/source
  (`discover.replace_existing=true`). Pass `--keep-existing` to retain historical runs.
* `--meme-mod` applies to MEME only; use it when each sequence is expected to contain one site.
* `--meme-prior` applies to MEME only; `addone` is a good default for sparse site sets.
* Use `--tool-path` or the `MEME_BIN` environment variable to point at a specific install.
  Relative `--tool-path` values resolve from the workspace root.
* MEME Suite is a system dependency; install `streme`/`meme` via your system package manager,
  pixi, or the official MEME Suite installer, and ensure they are discoverable.
  If you use the repo's pixi toolchain, run `pixi run cruncher -- discover ...` so MEME is on PATH
  (place `-c/--config` after the subcommand when using pixi tasks).
* See [MEME Suite dependency guide](../guides/meme_suite.md) for a reproducible setup pattern.

---

#### `cruncher doctor`

Fail-fast environment checks for external dependencies (currently MEME Suite).

Inputs:

* optional CONFIG (explicit or resolved)

Network:

* no (local)

Examples:

* `cruncher doctor <config>`
* `cruncher doctor --tool streme --tool-path /opt/meme/bin <config>`

---

#### `cruncher targets`

Check readiness for the configured `regulator_sets` (or a category preview).

Inputs:

* CONFIG (explicit or resolved)

Network:

* no (catalog + config only)

Subcommands:

* `targets list`
* `targets status`
* `targets candidates`
* `targets stats`

Examples:

* `cruncher targets status <config>`
* `cruncher targets candidates --fuzzy <config>`
* `cruncher targets list --category Category2 <config>`

---

#### `cruncher sources`

List or inspect ingestion sources.

Inputs:

* optional CONFIG (explicit or resolved)

Network:

* `sources list` is local-only
* `sources datasets` and `sources summary --scope remote|both` contact upstream services

Subcommands:

* `sources list [config]` — list registered sources (auto-detects config in CWD to include local sources; pass CONFIG when elsewhere)
* `sources info <source> [config]`
* `sources datasets <source> [config]` — list HT datasets (if supported)
* `sources summary [config]` — summarize cache + remote inventory (supports JSON output, combined view)

Example:

* `cruncher sources list configs/config.yaml`
* `cruncher sources datasets regulondb configs/config.yaml --tf lexA`
* `cruncher sources summary configs/config.yaml`
* `cruncher sources summary --view combined configs/config.yaml`
* `cruncher sources summary --scope remote --format json configs/config.yaml`
* `cruncher sources summary --json-out summary.json configs/config.yaml`

Regulator inventory for a single source:

* `cruncher sources summary --source regulondb --scope cache configs/config.yaml`
* `cruncher sources summary --source regulondb --scope remote --remote-limit 200 configs/config.yaml`
* `cruncher sources summary --source regulondb --view combined configs/config.yaml`

Note:

* `sources list` attempts full config resolution (workspace/CWD). If none is found, it lists built-in sources only.
  Pass CONFIG (or set `CRUNCHER_CONFIG`/`CRUNCHER_WORKSPACE`) to include local sources from a workspace config.
* Some sources do not expose full remote inventories; use `--remote-limit` (partial counts)
  or `--scope cache` if you only need cached regulators.
* `sources datasets --dataset-source <X>` performs a strict row-level source filter on returned datasets.

Example output (cache, abridged; captured with `CRUNCHER_LOG_LEVEL=WARNING` and `COLUMNS=200`):

```bash
        Cache overview
      (source=regulondb)
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Metric            ┃ Value   ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ entries           │ 2       │
│ sources           │ 1       │
│ TFs               │ 2       │
│ motifs            │ 0       │
│ site sets         │ 2       │
│ sites (seq/total) │ 203/203 │
│ datasets          │ 0       │
└───────────────────┴─────────┘
                  Cache by source (source=regulondb)
┏━━━━━━━━━━━┳━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Source    ┃ TFs ┃ Motifs ┃ Site sets ┃ Sites (seq/total) ┃ Datasets ┃
┡━━━━━━━━━━━╇━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ regulondb │   2 │      0 │         2 │ 203/203           │        0 │
└───────────┴─────┴────────┴───────────┴───────────────────┴──────────┘
                  Cache regulators (source=regulondb)
┏━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ TF   ┃ Sources   ┃ Motifs ┃ Site sets ┃ Sites (seq/total) ┃ Datasets ┃
┡━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ cpxR │ regulondb │      0 │         1 │ 154/154           │        0 │
│ lexA │ regulondb │      0 │         1 │ 49/49             │        0 │
└──────┴───────────┴────────┴───────────┴───────────────────┴──────────┘
```

---

#### `cruncher cache`

Inspect cache integrity.

Inputs:

* CONFIG (explicit or resolved)

Network:

* no (cache only)

* `cache stats <config>` — counts of cached motifs and site sets
* `cache verify <config>` — verify cache paths exist on disk
* `cache clean <config>` — list generated `__pycache__` / `.pytest_cache` directories (dry-run by default; use `--apply` to delete)
  * default scan scope is package root (`src/dnadesign/cruncher/`)
  * use `--scope workspace|package|repo` to change scan scope
  * use `--root <dir>` to scan an explicit directory (overrides `--scope`)

---

#### `cruncher status`

Bird’s-eye view of cache, targets, and recent runs.

Inputs:

* CONFIG (explicit or resolved)

Network:

* no (cache + run index only)

Example:

* `cruncher status <config>`
* `cruncher status --runs 10 <config>`

---

#### `cruncher runs`

Inspect past run artifacts.

Inputs:

* CONFIG (explicit or resolved)
* run name or run directory path for `show/watch`

Network:

* no (run artifacts only)

* `runs list <config>` — list run folders (optionally filter by stage).
* `runs show <config> <run>` — show manifest + artifacts (run name or run dir)
* `runs latest <config> --set-index 1` — print most recent run for a regulator set
* `runs best <config> --set-index 1` — print best run by `best_score` for a regulator set
* `runs watch <config> <run>` — live progress snapshot (run name or run dir; reads `meta/run_status.json`, optionally `optimize/state/metrics.jsonl`)
* `runs rebuild-index <config>` — rebuild `<workspace>/.cruncher/run_index.json`
* `runs repair-index <config>` — validate and optionally remove index entries missing run directories/manifests (`--apply`)
* `runs clean <config> --stale` — mark stale `running` runs as `aborted` (use `--drop` to remove from the index)
* `runs prune <config>` — archive old runs under `<out_dir>/_archive/<stage>/<YYYY-MM>/` with deterministic retention (`--keep-latest`, `--older-than-days`; dry-run unless `--apply`)
  * use `--repair-index` to drop invalid run-index entries before pruning
  * without `--repair-index`, prune requires a valid run index and exits with an actionable repair command

Tip: inside a workspace you can drop the config argument entirely (for example,
`cruncher runs show <run>` or `cruncher runs list`).

Notes:
* `runs watch --plot` writes a live PNG plot to `<run_dir>/live/live_metrics.png`.
* `runs watch --metric-points` and `--metric-width` control the trend window size.
* `runs watch --plot-path` writes plots to a custom path; `--plot-every` controls refresh cadence.

---

#### `cruncher config`

Summarize effective configuration settings.

Inputs:

* optional config path (`--config/-c`), otherwise resolved from workspace/CWD

Network:

* no

Examples:

* `cruncher config`
* `cruncher config summary`
* `cruncher config summary <config>`
* `cruncher config --config <config>`

Note:

* you can pass `--config/-c` before or after the subcommand; if omitted, Cruncher
  resolves the config from the current directory.

---

#### `cruncher optimizers`

List available optimizer kernels.

Inputs:

* none

Network:

* no

Example:

* `cruncher optimizers list`

Note:

* Cruncher defaults to `gibbs_anneal`; this list is informational for kernel development.

---

#### Global options

* `--log-level INFO|DEBUG|WARNING` (or set `CRUNCHER_LOG_LEVEL`)
* `--config/-c <path>` (or set `CRUNCHER_CONFIG`) to pin a specific config file
* `--workspace/-w <name|index|path>` (or set `CRUNCHER_WORKSPACE`) to pick a workspace config


---

@e-south
