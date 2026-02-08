# Cruncher architecture

## Contents
- [Cruncher architecture](#cruncher-architecture)
- [Run lifecycle](#run-lifecycle)
- [Layers and responsibilities](#layers-and-responsibilities)
- [On-disk layout](#on-disk-layout)
- [Run artifacts](#run-artifacts)
- [Reproducibility boundaries](#reproducibility-boundaries)
- [Extensibility points](#extensibility-points)
- [Related docs](#related-docs)

This doc describes the Cruncher run lifecycle, module boundaries, and on-disk artifacts.

### Run lifecycle

1. **fetch** -> cache motifs/sites and update `catalog.json`
2. **lock** -> resolve TFs to exact cached artifacts (`<workspace>/.cruncher/locks/<config>.lock.json`)
3. **parse** *(optional)* -> validate locked PWMs and refresh the parse cache in workspace state (no logo rendering)
4. **sample** -> run MCMC and write sequences/trace + manifests
5. **analyze** -> curated `plot__*`/`table__*` artifacts + report from sample artifacts (offline, written into the run directory)

---

### Layers and responsibilities

Core contract:

- **Network access is explicit** (fetch and remote inventory).
- The **store** is the only persistence layer (project-local).
- The **core** (PWM scoring + optimizers) is pure compute (no I/O).
- **Analyze** reads run artifacts only and can run offline.

#### `core/` (pure compute)
- PWM representation and validation
- scoring / evaluator logic
- sequence state and move operators
- optimizer kernels (parallel tempering)
- No I/O (no filesystem, no network)

#### `ingest/` (ports/adapters)
- source adapters (RegulonDB first)
- normalization into canonical records (motifs + sites)
- optional hydration (coordinates -> sequences) via genome providers

#### `store/` (local persistence)
- the on-disk catalog (what's cached)
- lockfiles (what's pinned)
- run index (what runs exist)

#### `analysis/` (analysis + diagnostics)
- plot registry, per-PWM summaries, and analysis helpers
- plot implementations live under `analysis/plots/`

#### `artifacts/` (run layout + manifests)
- run directory layout + status helpers
- manifest + artifact bookkeeping utilities

#### `viz/` (plotting)
- matplotlib/logomaker setup
- PWM logo rendering + visualization helpers

#### `integrations/` (external tools)
- wrappers for external binaries (e.g., MEME Suite)

#### `app/` (orchestration)
- fetch / lock / parse / sample / analyze coordination
- translates CLI intent + config into concrete runs and artifacts

#### `cli/` (UX only)
- Typer commands
- argument parsing, output formatting
- delegates work to app modules (no business logic)

---

### On-disk layout

**cruncher** uses **project-local state** (relative to the config file). Data artifacts live in the workspace;
tooling caches stay within the repo/workspace unless you override their environment variables.

Recommended workspace layout:

```
<workspace>/
config.yaml
.cruncher/
outputs/
```

In this repo, the bundled demo workspaces live at:

- `src/dnadesign/cruncher/workspaces/demo_basics_two_tf/`
- `src/dnadesign/cruncher/workspaces/demo_campaigns_multi_tf/`

#### Catalog root (`catalog.root`, default: `.cruncher/`)

```
<catalog.root>/
catalog.json
normalized/
motifs/<source>/<motif_id>.json
sites/<source>/<motif_id>.jsonl
genomes/              # if genome hydration is enabled
discoveries/          # MEME/STREME discovery runs
.mplcache/            # Matplotlib cache (MPLCONFIGDIR)
```

- `catalog.json` is the "what do we have cached?" index.
- `catalog.root` can be absolute or relative to the cruncher root (`src/dnadesign/cruncher`); relative paths must not include `..`.
- By default the catalog cache is shared across workspaces (`src/dnadesign/cruncher/.cruncher`); locks/run_index live in each workspace's `.cruncher/`.

#### Workspace state (per workspace `.cruncher/`)

```
<workspace>/.cruncher/
locks/<config>.lock.json
run_index.json
parse/latest/input/{lockfile.json,parse_manifest.json,pwm_summary.json}
parse/previous/input/{lockfile.json,parse_manifest.json,pwm_summary.json}
```

- `locks/<config>.lock.json` pins TF names -> exact cached artifacts + hashes.
- `run_index.json` tracks run folders for `cruncher runs ...` within that workspace.
- `parse/{latest,previous}` stores parse-stage validation artifacts outside user-facing sample outputs.

#### Tooling caches

- Matplotlib writes its cache under `<catalog.root>/.mplcache/` unless `MPLCONFIGDIR` is set.
- Numba JIT cache defaults to `<workspace>/.cruncher/numba_cache` unless `NUMBA_CACHE_DIR` is set.

#### Run outputs (`out_dir`, e.g. `outputs/`)

Each regulator set gets a retention-oriented **run slot** with two pointers:

- `latest/` - most recent run for that set
- `previous/` - immediately prior run for that set (if any)

Slot roots:

- single regulator set: `<workspace>/<out_dir>/latest/` and `<workspace>/<out_dir>/previous/`
- multiple regulator sets: `<workspace>/<out_dir>/setN_<tf-slug>/latest/` and `.../previous/`

Within each run directory, Cruncher uses a stable, stage-agnostic subdirectory layout:

```
<run_dir>/
  input/
  optimize/
  output/
  plots/
  run_manifest.json
  run_status.json
  config_used.yaml
```

---

### Run artifacts

A typical **sample** run directory contains:

- `run_manifest.json`, `run_status.json`, `config_used.yaml` - run metadata + status
- `input/lockfile.json` - pinned input snapshot (reproducibility boundary)
- `optimize/sequences.parquet`, `optimize/trace.nc`, `optimize/elites*`, `optimize/random_baseline*` - sampling outputs
- `metrics.jsonl` - live sampling metrics (if enabled)
- `output/summary.json` - canonical analysis summary
- `output/report.json` + `output/report.md` - analysis report outputs from `cruncher analyze`
- `output/plot_manifest.json` + `output/table_manifest.json` + `output/manifest.json` - analysis inventories
- `plots/plot__*` - curated plot outputs
- `output/table__*` - curated table outputs

---

### Reproducibility boundaries

- **Lockfiles are mandatory** for `parse` and `sample`.
- If you change inputs that affect TF resolution (e.g., PWM source, site filters, dataset selection),
  **re-lock** so the lockfile hash set matches reality.
- `analyze` validates the lockfile recorded in the run manifest.

---

### Extensibility points

- **Sources:** add a new adapter under `ingest/adapters/` and register it in the source registry.
- **Local sources:** configure `ingest.local_sources` for local motif directories (no new code required).
- **Parsers:** add a parser under `io/parsers/` or register via `io.parsers.extra_modules`.
- **Optimizers:** add a new kernel and register it in `core/optimizers/registry.py`.
- **Analysis plots:** add a plot implementation and register it in the analysis plot registry.

---

### Related docs

- [Intent + lifecycle](../guides/intent_and_lifecycle.md)
- [Config reference](config.md)

@e-south
