## cruncher CLI

Most commands operate relative to a `config.yaml` file. For an end-to-end walkthrough, see the [demo](demo.md).

### Contents

1. [Core lifecycle commands](#core-lifecycle-commands)
2. [Discovery and inspection](#discovery-and-inspection)

---

### Core lifecycle commands

#### `cruncher fetch motifs`

Caches motif matrices into `<catalog_root>/normalized/motifs/...`.

When to use:

* you want `cruncher.motif_store.pwm_source: matrix`
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

---

#### `cruncher fetch sites`

Caches binding-site instances into `<catalog_root>/normalized/sites/...`.

When to use:

* you want `cruncher.motif_store.pwm_source: sites`
* you want curated or HT site sets cached locally
* you need hydration for coordinate-only peaks

Examples:

* `cruncher fetch sites --tf lexA --tf cpxR <config>`
* `cruncher fetch sites --dry-run --tf lexA <config>`
* `cruncher fetch sites --dataset-id <id> --tf lexA <config>`
* `cruncher fetch sites --genome-fasta genome.fna <config>`

Common options:

* `--tf`, `--motif-id`, `--dataset-id`, `--limit`
* `--hydrate` (hydrates missing sequences)
* `--offline`, `--update`
* `--genome-fasta`
* `--summary/--no-summary`, `--paths`

Note:

* `--hydrate` with no `--tf/--motif-id` hydrates all cached site sets by default.

---

#### `cruncher lock`

Resolves TF names in `cruncher.regulator_sets` to exact cached artifacts (IDs + hashes).
Writes `<catalog_root>/locks/<config>.lock.json`.

When to use:

* before `parse` and `sample`
* whenever you change anything affecting TF resolution (PWM source, site kinds, dataset selection, etc.)

Example:

* `cruncher lock <config>`

---

#### `cruncher parse`

Validates cached PWMs (matrix- or site-derived) and renders logos/QC artifacts.

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

Examples:

* `cruncher analyze --latest <config>`
* `cruncher analyze --run <run_name> <config>`
* `cruncher analyze --tf-pair lexA,cpxR <config>`
* `cruncher analyze --plots trace --plots score_hist <config>`
* `cruncher analyze --list-plots`

Preconditions:

* provide runs via `analysis.runs`, `--run`, or `--latest`
* trace-dependent plots require `trace.nc`

---

#### `cruncher report`

Writes `report.json` and `report.md` for a sample run.

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

Example:

* `cruncher notebook <path/to/sample_run> --latest`

Notes:

* requires `marimo` to be installed (for example: `uv add --group notebooks marimo`)
* useful when you want interactive slicing/filtering beyond static plots
* strict by default: requires `summary.json` + `plot_manifest.json` to exist and parse, and `summary.json` must include a non-empty `tf_names` list
* pass `--lenient` to generate anyway (warnings appear in the Overview tab)
* when `summary.json` is missing, lenient mode falls back to `analysis/` as an unindexed entry
* plot output status is refreshed from disk so missing files are shown accurately
* text outputs (for example, `diag__convergence.txt`) render inline in the Plots tab
* if running in lenient mode and `summary.json` lacks `tf_names`, scatter controls are disabled with an inline warning
* the notebook includes:
  * Overview tab with run metadata and explicit warnings for missing/invalid analysis artifacts
  * Tables tab with a Top-K slider and a per-PWM data explorer
  * Plots tab with TF dropdowns, chain/draw range controls, and inline plot previews

---

### Discovery and inspection

#### `cruncher catalog`

Inspect cached motifs and site sets.

Subcommands:

* `catalog list` — list cached motifs and site sets
* `catalog search` — search by TF name or motif ID
* `catalog resolve` — resolve a TF name to cached candidates
* `catalog show` — show metadata for a cached `<source>:<motif_id>`

Examples:

* `cruncher catalog list <config>`
* `cruncher catalog search <config> lexA --fuzzy`
* `cruncher catalog show <config> regulondb:RDBECOLITFC00214`

---

#### `cruncher targets`

Check readiness for the configured `regulator_sets`.

Subcommands:

* `targets list`
* `targets status`
* `targets candidates`
* `targets stats`

Examples:

* `cruncher targets status <config>`
* `cruncher targets candidates --fuzzy <config>`

---

#### `cruncher sources`

List or inspect ingestion sources.

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

* `cache stats <config>` — counts of cached motifs and site sets
* `cache verify <config>` — verify cache paths exist on disk

---

#### `cruncher status`

Bird’s-eye view of cache, targets, and recent runs.

Example:

* `cruncher status <config>`

---

#### `cruncher runs`

Inspect past run artifacts.

* `runs list <config>` — list run folders (optionally filter by stage)
* `runs show <config> <run>` — show manifest + artifacts
* `runs latest <config>` — print most recent run
* `runs watch <config> <run>` — live progress snapshot
* `runs rebuild-index <config>` — rebuild `<catalog_root>/run_index.json`

Tip: inside a workspace you can drop the config argument entirely (for example,
`cruncher runs show <run>` or `cruncher runs list`).

---

#### `cruncher config`

Summarize effective configuration settings.

Examples:

* `cruncher config <config>`
* `cruncher config <config> summary`

Note:

* you can pass `--config/-c` before or after the subcommand; if omitted, Cruncher
  resolves the config from the current directory.

---

#### `cruncher optimizers`

List available optimizer kernels.

Example:

* `cruncher optimizers list`

---

### Global options

* `--log-level INFO|DEBUG|WARNING` (or set `CRUNCHER_LOG_LEVEL`)


---

@e-south
