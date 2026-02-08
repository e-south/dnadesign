# Cruncher CLI

## Contents
- [Cruncher CLI](#cruncher-cli)
- [Workspace discovery and config resolution](#workspace-discovery-and-config-resolution)
- [Quick command map](#quick-command-map)
- [Core lifecycle commands](#core-lifecycle-commands)
- [Discovery and inspection](#discovery-and-inspection)
- [Global options](#global-options)

This reference summarizes the Cruncher CLI surface, grouped by lifecycle stage and workflow.

> **Intent:** Cruncher is an optimization engine for **fixed-length** multi-TF PWM sequence design that returns a **diverse elite set** - not posterior inference.
>
> **When to use:** design under tight length constraints; explore motif compatibility tradeoffs; generate a small candidate set for assays; sweep many regulator sets via campaigns + summarize results.

### Workspace discovery and config resolution

Cruncher resolves config from `--config/-c` or `--workspace/-w`, then the nearest `config.yaml` in the current directory or parents, then known workspace roots. If multiple workspaces are found, **cruncher** prompts for a selection (interactive shells only).

See available workspaces with:

```
cruncher workspaces list
```

---

### Quick command map

* **Cache data** → `fetch motifs` / `fetch sites`
* **Inspect cache** → `sources ...` / `catalog ...`
* **Pin TFs** → `lock`
* **Validate motifs** → `parse`
* **Render logos** → `catalog logos`
* **Optimize** → `sample`
* **Analyze** → `analyze`, `notebook`
* **Campaigns** → `campaign validate|generate|summarize|notebook`
* **Run management** → `runs list/show/latest/best/watch/clean`
* **Workspace health** → `status`

---

### Core lifecycle commands

#### `cruncher fetch motifs`

Caches motif matrices into `<catalog.root>/normalized/motifs/...`.

Inputs:

* CONFIG (explicit or resolved)
* at least one of `--tf`, `--motif-id`, or `--campaign`

Network:

* yes by default; use `--offline` to restrict to cached motifs only

When to use:

* you want `cruncher.catalog.pwm_source: matrix`
* you want to reuse alignment/matrix payloads across runs

Examples:

* `cruncher fetch motifs --tf lexA --tf cpxR <config>`
* `cruncher fetch motifs --motif-id RDBECOLITFC00214 <config>`
* `cruncher fetch motifs --source omalley_ecoli_meme --tf lexA <config>`
* `cruncher fetch motifs --campaign regulators_v1 <config>`
* `cruncher fetch motifs --dry-run --tf lexA <config>`

Common options:

* `--tf`, `--motif-id`, `--campaign`, `--source`
* `--apply-selectors/--no-selectors` (campaigns)
* `--dry-run`, `--all`, `--offline`, `--update`
* `--summary/--no-summary`, `--paths`

Outputs:

* writes cached motif JSON files and updates `catalog.json`
* prints a summary table by default (or raw paths with `--paths`)

Note:

* `--campaign` applies campaign selectors by default; use `--no-selectors` to fetch raw category TFs when the
  local catalog is empty.
* `--source` defaults to the first available entry in `catalog.source_preference` (skipping entries that are
  not registered ingest sources); if the list is empty or none are available you must pass `--source` explicitly.

---

#### `cruncher fetch sites`

Caches binding-site instances into `<catalog.root>/normalized/sites/...`.

Inputs:

* CONFIG (explicit or resolved)
* at least one of `--tf`, `--motif-id`, `--campaign`, or `--hydrate`

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
* `cruncher fetch sites --campaign regulators_v1 <config>`

Common options:

* `--tf`, `--motif-id`, `--campaign`, `--dataset-id`, `--limit`, `--source`
* `--apply-selectors/--no-selectors` (campaigns)
* `--hydrate` (hydrates missing sequences)
* `--offline`, `--update`
* `--genome-fasta`
* `--summary/--no-summary`, `--paths`

Outputs:

* writes cached site JSONL files and updates `catalog.json`
* prints a summary table by default (or raw paths with `--paths`)

Note:

* `--hydrate` with no `--tf/--motif-id` hydrates all cached site sets by default.
* `--campaign` applies campaign selectors by default; use `--no-selectors` to fetch raw category TFs when the
  local catalog is empty.
* `--source` defaults to the first available entry in `catalog.source_preference` (skipping entries that are
  not registered ingest sources); if the list is empty or none are available you must pass `--source` explicitly.

---

#### `cruncher lock`

Resolves TF names in `workspace.regulator_sets` (under the root `cruncher:` block) to exact cached artifacts (IDs + hashes).
Writes `<workspace>/.cruncher/locks/<config>.lock.json`.

Inputs:

* CONFIG (explicit or resolved)
* cached motifs/sites for the configured regulators

Network:

* no (cache-only)

When to use:

* before `parse` and `sample`
* whenever you change anything affecting TF resolution (PWM source, site kinds, dataset selection, etc.)

Example:

* `cruncher lock <config>`

---

#### `cruncher campaign generate`

Expands a campaign into explicit `regulator_sets` and writes a derived config plus a manifest.

Inputs:

* CONFIG (explicit or resolved)
* `--campaign <name>` matching a campaign in config

Network:

* no (uses cache only if selectors require metrics)

Example:

* `cruncher campaign generate --campaign regulators_v1 --out config.campaign.yaml <config>`

Notes:

* The base config must define `regulator_categories` and `campaigns`.
* Selector filters require cached motifs/sites; fetch before generating if you use them.
* The manifest is written alongside the output config by default.
* `--out` must live alongside the workspace `config.yaml` so `out_dir` remains workspace-relative.
  Relative `--out` paths are interpreted from the workspace root.

---

#### `cruncher campaign validate`

Validate a campaign against cached motifs/sites and selector rules.

Inputs:

* CONFIG (explicit or resolved)
* `--campaign <name>`

Network:

* no (cache-only; `--no-metrics` avoids cache requirements)

Examples:

* `cruncher campaign validate --campaign regulators_v1 <config>`
* `cruncher campaign validate --campaign regulators_v1 --no-selectors <config>`
* `cruncher campaign validate --campaign regulators_v1 --show-filtered <config>`

Notes:

* `--no-selectors` disables selector filtering; add `--no-metrics` to validate categories without cached data.
* `--metrics` (default) computes per-TF metrics and requires a local catalog.

---

#### `cruncher campaign summarize`

Aggregates multiple campaign runs into summary tables and plots (offline).

Inputs:

* CONFIG (explicit or resolved)
* `--campaign <name>`
* `--runs` (optional; defaults to all sample runs)

Network:

* no (uses local run artifacts; `--metrics` requires local catalog)

Examples:

* `cruncher campaign summarize --campaign regulators_v1 --runs outputs/* --config <config>`
* `cruncher campaign summarize --campaign regulators_v1 --no-metrics <config>`

Outputs:

* `campaign_summary.csv`, `campaign_best.csv`
* plots under the campaign output root (e.g., `plot__best_jointscore_bar.png`,
  `plot__tf_coverage_heatmap.png`, `plot__joint_trend.png`, `plot__pareto_projection.png`)

Notes:

* `--metrics` requires a local catalog; fetch motifs/sites first.
* When `--runs` is omitted, stale sample run-index entries are auto-repaired before summary.
* `--skip-missing` skips runs missing required `table_manifest.json` entries/files for
  `scores_summary` and `metrics_joint` (typically `table__scores_summary.parquet` and
  `table__metrics_joint.parquet`).
* With site-derived PWMs, `--metrics` also requires `catalog.site_window_lengths`
  for TFs with variable site lengths. Use `--no-metrics` if you haven't set them.

---

#### `cruncher campaign notebook`

Generates a marimo notebook for exploring `campaign_summary.csv` outputs.

Inputs:

* CONFIG (explicit or resolved)
* `--campaign <name>`

Network:

* no (requires local summary outputs; marimo is a local dependency)

Examples:

* `cruncher campaign notebook --campaign regulators_v1 <config>`
* `cruncher campaign notebook --campaign regulators_v1 --out outputs/campaign/regulators_v1/latest <config>`

Notes:

* Requires `cruncher campaign summarize` to have been run first (summary CSVs + manifest present).
* Install marimo with `uv sync --locked --group notebooks`.

---

#### `cruncher parse`

Validates cached PWMs (matrix- or site-derived) and writes parse-cache artifacts in workspace state.

Inputs:

* CONFIG (explicit or resolved)
* lockfile (from `cruncher lock`)

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
* Parse is idempotent for identical inputs; if matching outputs already exist, it reports
  the existing run instead of creating a new one.
* Parse artifacts live under `<workspace>/.cruncher/parse/{latest,previous}/input/` and are intentionally
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
* `cruncher sample --debug <config>`

Precondition:

* lockfile exists (`cruncher lock <config>`)

Notes:

* `sample.output.save_sequences: true` is required for later analysis.
* `sample.output.save_trace: true` enables trace-based diagnostics.
* `sample.output.save_trace: false` skips ArviZ trace construction and reduces sample runtime/memory overhead.
* Sampling uses the internal parallel tempering kernel; there is no optimizer selection in the config.
* `--verbose` enables periodic progress logging; `--debug` enables very verbose debug logs.

---

#### `cruncher analyze`

Generates diagnostics and plots for one or more sample runs.

Inputs:

* CONFIG (explicit or resolved)
* runs via `analysis.run_selector`/`analysis.runs` or `--run` (defaults to latest sample run if empty)
* run artifacts: `optimize/sequences.parquet` (required), `optimize/elites.parquet` (required),
  `optimize/elites_hits.parquet` (required), `optimize/random_baseline.parquet` (required),
  and `optimize/trace.nc` for trace-based plots

Network:

* no (run artifacts only)

Examples:

* `cruncher analyze --latest <config>`
* `cruncher analyze --run <run_name|run_dir> <config>`
* `cruncher analyze --summary <config>`

Preconditions:

* provide runs via `analysis.runs`/`--run` or rely on the default latest run
* trace-dependent plots require `optimize/trace.nc`
* if `<run_dir>/output/` exists without `summary.json`, remove the incomplete `output/` folder before re-running `cruncher analyze`
* each sample run snapshots the lockfile under `input/lockfile.json`; analysis uses that snapshot to avoid mismatch if the workspace lockfile changes later

Outputs:

* tables: `output/table__scores_summary.parquet`, `output/table__elites_topk.parquet`,
  `output/table__metrics_joint.parquet`, `output/table__opt_trajectory_points.parquet`,
  `output/table__diagnostics_summary.json`, `output/table__objective_components.json`,
  `output/table__elites_mmr_summary.parquet`, `output/table__elites_nn_distance.parquet`
* plots: `plots/plot__run_summary.<plot_format>`, `plots/plot__opt_trajectory.<plot_format>`,
  `plots/plot__elites_nn_distance.<plot_format>`, `plots/plot__overlap_panel.<plot_format>`,
  `plots/plot__health_panel.<plot_format>` (trace only)
* reports: `output/report.json`, `output/report.md`
* summaries: `output/summary.json`, `output/manifest.json`, `output/plot_manifest.json`, `output/table_manifest.json`

Note:

* Analyze rewrites the latest analysis outputs each run; set `analysis.archive=true` to keep prior reports.
* Use `cruncher analyze --summary` to print the highlights from `report.json`.

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

* requires `marimo` to be installed (for example: `uv add --group notebooks marimo`)
* useful when you want interactive slicing/filtering beyond static plots
* strict artifact contract: requires `output/summary.json`, `output/plot_manifest.json`, and `output/table_manifest.json` to exist and parse, `output/summary.json` must include a non-empty `tf_names` list, and `output/table_manifest.json` must provide `scores_summary`, `metrics_joint`, and `elites_topk` entries with existing files
* plot output status is refreshed from disk so missing files are shown accurately
* the Refresh button re-scans analysis entries and updates plot/table status without restarting marimo
* the notebook infers `run_dir` from its location; keep it under `<run_dir>/` or regenerate it
* plots are loaded from `output/plot_manifest.json`; the curated keys are `run_summary`, `opt_trajectory`, `elites_nn_distance`, plus optional `overlap_panel` and `health_panel` entries when generated
* the notebook includes:
  * Overview tab with run metadata and explicit warnings for missing/invalid analysis artifacts
  * Tables tab with a Top-K slider and per-table previews from `output/table_manifest.json`
  * Plots tab with inline previews and generated/skipped status from `output/plot_manifest.json`

---

### Discovery and inspection

#### `cruncher workspaces`

List discoverable workspaces and their config paths.

Inputs:

* none

Network:

* no

Example:

* `cruncher workspaces list`

---

#### `cruncher catalog`

Inspect cached motifs and site sets.

Use `catalog pwms` to compute PWMs from cached matrices or binding sites and
survey their lengths/bit scores (including any configured PWM window), and `catalog logos` to render PNG logos for the
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
under `outputs/logos/catalog/`, it reports the existing path instead of writing a new run.

Examples:

* `cruncher catalog list <config>`
* `cruncher catalog search <config> lexA --fuzzy`
* `cruncher catalog show <config> regulondb:RDBECOLITFC00214`
* `cruncher catalog pwms <config>`
* `cruncher catalog pwms --set 1 <config>`
* `cruncher catalog export-sites --set 1 --out densegen/sites.csv <config>`
* `cruncher catalog export-sites --set 1 --densegen-workspace demo_meme_three_tfs <config>`
* `cruncher catalog export-densegen --set 1 --out densegen/pwms <config>`
* `cruncher catalog export-densegen --set 1 --densegen-workspace demo_meme_three_tfs <config>`
* `cruncher catalog logos --set 1 <config>`

`catalog export-densegen` and `catalog export-sites` accept `--densegen-workspace` (workspace
name under `src/dnadesign/densegen/workspaces/` or an absolute path). When provided, outputs
default to the workspace `inputs/` locations and must stay within that directory.
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
* If `--minw/--maxw` are omitted (and unset in config), Cruncher derives them from the min/max
  site lengths per TF.
* Use `cruncher targets stats` to set `--minw/--maxw` from site-length ranges.
* If you plan to run both MEME and STREME, set distinct `discover.source_id` values between runs to avoid lock ambiguity.
  You can also pass `--source-id` per run to avoid editing config.
* By default discovery replaces previous discovered motifs for the same TF/source
  (`discover.replace_existing=true`). Pass `--keep-existing` to retain historical runs.
* `--meme-mod` applies to MEME only; use it when each sequence is expected to contain one site.
* `--meme-prior` applies to MEME only; `addone` is a good default for sparse site sets.
* Use `--tool-path` or the `MEME_BIN` environment variable to point at a specific install.
  Relative `--tool-path` values resolve from the config file location.
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

Check readiness for the configured `regulator_sets` (or a category/campaign preview).

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
* `cruncher targets status --campaign regulators_v1 <config>`

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

* `cruncher sources list config.yaml`
* `cruncher sources datasets regulondb config.yaml --tf lexA`
* `cruncher sources summary config.yaml`
* `cruncher sources summary --view combined config.yaml`
* `cruncher sources summary --scope remote --format json config.yaml`
* `cruncher sources summary --json-out summary.json config.yaml`

Regulator inventory for a single source:

* `cruncher sources summary --source regulondb --scope cache config.yaml`
* `cruncher sources summary --source regulondb --scope remote --remote-limit 200 config.yaml`
* `cruncher sources summary --source regulondb --view combined config.yaml`

Note:

* `sources list` attempts full config resolution (workspace/CWD). If none is found, it lists built-in sources only.
  Pass CONFIG (or set `CRUNCHER_CONFIG`/`CRUNCHER_WORKSPACE`) to include local sources from a workspace config.
* Some sources do not expose full remote inventories; use `--remote-limit` (partial counts)
  or `--scope cache` if you only need cached regulators.

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
* `runs watch <config> <run>` — live progress snapshot (run name or run dir; reads `run_status.json`, optionally `metrics.jsonl`)
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

* CONFIG (explicit or resolved)

Network:

* no

Examples:

* `cruncher config <config>`
* `cruncher config <config> summary`

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

* Cruncher uses parallel tempering internally; this list is informational for kernel development.

---

### Global options

* `--log-level INFO|DEBUG|WARNING` (or set `CRUNCHER_LOG_LEVEL`)
* `--config/-c <path>` (or set `CRUNCHER_CONFIG`) to pin a specific config file
* `--workspace/-w <name|index|path>` (or set `CRUNCHER_WORKSPACE`) to pick a workspace config


---

@e-south
