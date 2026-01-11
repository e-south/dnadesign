## cruncher CLI

Most commands operate relative to a `config.yaml` file. You can also pass `--config/-c` globally (before the command). Some commands (notably `cruncher notebook`) operate on a run directory instead of a config.

### Contents

1. [Workspace discovery and config resolution](#workspace-discovery-and-config-resolution)
2. [Which command should I use?](#which-command-should-i-use)
2. [Core lifecycle commands](#core-lifecycle-commands)
3. [Discovery and inspection](#discovery-and-inspection)
4. [Global options](#global-options)

> Network access occurs in fetch and remote inventory commands (for example, `cruncher fetch ...`, `cruncher sources summary --scope remote`, `cruncher sources datasets`). Some workflows (such as site hydration via NCBI) also use the network depending on your config. Use `--offline` where supported to force cache-only behavior.

---

### Workspace discovery and config resolution

Cruncher resolves the config path in this order:

1. `--config/-c` (global or per-command), positional `CONFIG` (explicit config path)
2. `--workspace/-w` (workspace name, index from `workspaces list`, or a path to a workspace dir/config)
   * also accepts `CRUNCHER_WORKSPACE=<name|index|path>`
3. `config.yaml` / `cruncher.yaml` in the current directory
4. walk up parent directories looking for a config
5. workspace discovery in known roots:
   * git root: `workspaces/*/config.yaml` (and `workspace/*/config.yaml`)
   * git root: `src/dnadesign/cruncher/workspaces/*/config.yaml` (bundled demo)
   * any roots in `CRUNCHER_WORKSPACE_ROOTS` (colon-separated)

If exactly one workspace is discovered, Cruncher auto-selects it and logs a one-line note.
If multiple are found, Cruncher prints a numbered list and shows how to select one via `--workspace` or `--config`.

To see what is available (with stable indices), run:

```
cruncher workspaces list
```

---

### Which command should I use?

Common tasks mapped to commands:

* **Get data into the cache** → `cruncher fetch motifs` (matrices) or `cruncher fetch sites` (binding sites / HT datasets).
* **Check what is available** → `cruncher sources list/summary` (source inventory) or `cruncher catalog list/search/show/pwms/logos` (cached entries + PWM/logo outputs).
* **Pin TFs for reproducible runs** → `cruncher lock`.
* **Validate cached PWMs + render logos** → `cruncher parse`.
* **Run the optimizer** → `cruncher sample`.
* **Analyze a sample run** → `cruncher analyze` (plots/tables), then `cruncher report` (summary), optionally `cruncher notebook`.
* **Campaign workflows** → `cruncher campaign validate` (preflight), `cruncher campaign generate` (derived config),
  `cruncher campaign summarize` (aggregate results), `cruncher campaign notebook` (campaign summary exploration).
* **Check target readiness** → `cruncher targets status` (and `targets candidates/stats` for deeper inspection).
* **Inspect run artifacts** → `cruncher runs list/show/latest/watch`.
* **Quick snapshot of workspace health** → `cruncher status`.
* **Inspect config resolution** → `cruncher config` (summary table).
* **List available optimizers** → `cruncher optimizers list`.
* **Find workspaces** → `cruncher workspaces list`.

---

### Core lifecycle commands

#### `cruncher fetch motifs`

Caches motif matrices into `<catalog_root>/normalized/motifs/...`.

Inputs:

* CONFIG (explicit or resolved)
* at least one of `--tf`, `--motif-id`, or `--campaign`

Network:

* yes by default; use `--offline` to restrict to cached motifs only

When to use:

* you want `cruncher.motif_store.pwm_source: matrix`
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

---

#### `cruncher fetch sites`

Caches binding-site instances into `<catalog_root>/normalized/sites/...`.

Inputs:

* CONFIG (explicit or resolved)
* at least one of `--tf`, `--motif-id`, `--campaign`, or `--hydrate`

Network:

* yes by default; use `--offline` to restrict to cached sites only

When to use:

* you want `cruncher.motif_store.pwm_source: sites`
* you want curated or HT site sets cached locally
* you need hydration for coordinate-only peaks

Examples:

* `cruncher fetch sites --tf lexA --tf cpxR <config>`
* `cruncher fetch sites --dry-run --tf lexA <config>`
* `cruncher fetch sites --dataset-id <id> --tf lexA <config>`
* `cruncher fetch sites --genome-fasta genome.fna <config>`
* `cruncher fetch sites --campaign regulators_v1 <config>`

Common options:

* `--tf`, `--motif-id`, `--campaign`, `--dataset-id`, `--limit`
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

---

#### `cruncher lock`

Resolves TF names in `cruncher.regulator_sets` to exact cached artifacts (IDs + hashes).
Writes `<catalog_root>/locks/<config>.lock.json`.

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

* `cruncher campaign summarize --campaign regulators_v1 --runs runs/* --config <config>`
* `cruncher campaign summarize --campaign regulators_v1 --no-metrics <config>`

Outputs:

* `campaign_summary.csv`, `campaign_best.csv`
* plots under `plots/` (including `best_jointscore_bar.png`, `tf_coverage_heatmap.png`,\n  `joint_trend.png`, and `pareto_projection.png`)

Notes:

* `--metrics` requires a local catalog; fetch motifs/sites first.
* `--skip-missing` skips runs missing `analysis/tables/joint_metrics.csv` or `score_summary.csv`.
* With site-derived PWMs, `--metrics` also requires `motif_store.site_window_lengths`
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
* `cruncher campaign notebook --campaign regulators_v1 --out runs/campaigns/<id> <config>`

Notes:

* Requires `cruncher campaign summarize` to have been run first (summary CSVs + manifest present).
* Install marimo with `uv sync --locked --group notebooks`.

---

#### `cruncher parse`

Validates cached PWMs (matrix- or site-derived) and renders logos/QC artifacts.

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

* `parse.plot.logo=false` skips logo rendering (still validates PWMs + writes a run manifest).
* When `motif_store.pwm_source=sites`, logos include a subtitle describing site provenance
  (curated, high-throughput, or combined).

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

Precondition:

* lockfile exists (`cruncher lock <config>`)

Notes:

* `cruncher.sample.save_sequences: true` is required for later analysis/reporting.
* `cruncher.sample.save_trace: true` enables trace-based diagnostics.

---

#### `cruncher analyze`

Generates diagnostics and plots for one or more sample runs.

Inputs:

* CONFIG (explicit or resolved)
* runs via `analysis.runs`, `--run`, or `--latest`
* run artifacts: `sequences.parquet` (required) and `trace.nc` for trace-based plots

Network:

* no (run artifacts only)

Examples:

* `cruncher analyze --latest <config>`
* `cruncher analyze --run <run_name> <config>`
* `cruncher analyze --tf-pair lexA,cpxR <config>`
* `cruncher analyze --plots trace --plots score_hist <config>`
* `cruncher analyze --list-plots`

Preconditions:

* provide runs via `analysis.runs`, `--run`, or `--latest`
* trace-dependent plots require `trace.nc`

Outputs:

* tables: `analysis/tables/score_summary.csv`, `analysis/tables/elite_topk.csv`, `analysis/tables/joint_metrics.csv`
* plots: `analysis/plots/score__pairgrid.png` (when `analysis.plots.pairgrid=true`)

---

#### `cruncher report`

Writes `report.json` and `report.md` for a sample run.

Inputs:

* CONFIG (explicit or resolved)
* run name (sample run)
* required artifacts: `sequences.parquet` and `trace.nc` (plus elites)

Network:

* no (run artifacts only)

Example:

* `cruncher report <config> <run_name>`

Preconditions:

* required artifacts must exist in the run directory (commonly `sequences.parquet`, `elites.parquet`, and often `trace.nc`)
* if required artifacts are missing, reporting should fail fast rather than silently omitting sections

Diagnostics note:

* R-hat requires at least 2 chains.
* ESS is not meaningful with too few draws.
  When metrics cannot be computed, reports should record diagnostics warnings.

Tip: if you are in a workspace with `config.yaml`, you can run `cruncher report <run_name>`
directly (or pass `--config` when running elsewhere).

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
* strict by default: requires `summary.json` + `plot_manifest.json` to exist and parse, and `summary.json` must include a non-empty `tf_names` list
* pass `--lenient` to generate anyway (warnings appear in the Overview tab)
* when `summary.json` is missing, lenient mode falls back to `analysis/` as an unindexed entry
* plot output status is refreshed from disk so missing files are shown accurately
* the Refresh button re-scans analysis entries and updates plot/table status without restarting marimo
* the notebook infers `run_dir` from its location; keep it under `<run_dir>/analysis/notebooks/` or regenerate it
* text outputs (for example, `diag__convergence.txt`) render inline in the Plots tab
* if running in lenient mode and `summary.json` lacks `tf_names`, scatter controls are disabled with an inline warning
* the notebook includes:
  * Overview tab with run metadata and explicit warnings for missing/invalid analysis artifacts
  * Tables tab with a Top-K slider and a per-PWM data explorer
  * Plots tab with TF dropdowns, chain/draw range controls, and inline plot previews

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
survey their lengths/bit scores, and `catalog logos` to render PNG logos for the
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
* `catalog logos` — render PWM logos for selected TFs or motif refs

Examples:

* `cruncher catalog list <config>`
* `cruncher catalog search <config> lexA --fuzzy`
* `cruncher catalog show <config> regulondb:RDBECOLITFC00214`
* `cruncher catalog pwms <config>`
* `cruncher catalog pwms --set 1 <config>`
* `cruncher catalog logos --set 1 <config>`

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

---

#### `cruncher status`

Bird’s-eye view of cache, targets, and recent runs.

Inputs:

* CONFIG (explicit or resolved)

Network:

* no (cache + run index only)

Example:

* `cruncher status <config>`

---

#### `cruncher runs`

Inspect past run artifacts.

Inputs:

* CONFIG (explicit or resolved)
* run name for `show/watch`

Network:

* no (run artifacts only)

* `runs list <config>` — list run folders (optionally filter by stage)
* `runs show <config> <run>` — show manifest + artifacts
* `runs latest <config>` — print most recent run
* `runs watch <config> <run>` — live progress snapshot (reads `run_status.json`, optionally `live_metrics.jsonl`)
* `runs rebuild-index <config>` — rebuild `<catalog_root>/run_index.json`

Tip: inside a workspace you can drop the config argument entirely (for example,
`cruncher runs show <run>` or `cruncher runs list`).

Notes:
* `runs watch --plot` writes a live PNG plot to `<run_dir>/live/live_metrics.png`.
* `runs watch --metric-points` and `--metric-width` control the trend window size.

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

---

### Global options

* `--log-level INFO|DEBUG|WARNING` (or set `CRUNCHER_LOG_LEVEL`)
* `--config/-c <path>` (or set `CRUNCHER_CONFIG`) to pin a specific config file
* `--workspace/-w <name|index|path>` (or set `CRUNCHER_WORKSPACE`) to pick a workspace config


---

@e-south
