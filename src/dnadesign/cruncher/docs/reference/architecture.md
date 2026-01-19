## cruncher architecture

**cruncher** is structured so that data access, sequence optimization logic, and the CLI can evolve independently.


### Run lifecycle

1. **fetch** → cache motifs/sites and update `catalog.json`
2. **lock** → resolve TFs to exact cached artifacts (`<workspace>/.cruncher/locks/<config>.lock.json`)
3. **parse** *(optional)* → validate cached PWMs (no logo rendering)
4. **sample** → run MCMC and write sequences/trace + manifests
5. **analyze** → plots/tables + report from sample artifacts (offline, written in analysis root)

---

### Layers and responsibilities

Core contract:

- **Network access is explicit** (fetch and remote inventory).
- The **store** is the only persistence layer (project‑local).
- The **core** (PWM scoring + optimizers) is pure compute (no I/O).
- **Analyze** reads run artifacts only and can run offline.

#### `core/` (pure compute)
- PWM representation and validation
- scoring / evaluator logic
- sequence state and move operators
- optimizer kernels (e.g., Gibbs, parallel tempering)
- No I/O (no filesystem, no network)

#### `ingest/` (ports/adapters)
- source adapters (RegulonDB first)
- normalization into canonical records (motifs + sites)
- optional hydration (coordinates → sequences) via genome providers

#### `store/` (local persistence)
- the on-disk catalog (what’s cached)
- lockfiles (what’s pinned)
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

#### Catalog root (`catalog_root`, default: `.cruncher/`)

```
<catalog_root>/
catalog.json
normalized/
motifs/<source>/<motif_id>.json
sites/<source>/<motif_id>.jsonl
genomes/              # if genome hydration is enabled
discoveries/          # MEME/STREME discovery runs
.mplcache/            # Matplotlib cache (MPLCONFIGDIR)
```

- `catalog.json` is the “what do we have cached?” index.
- `catalog_root` can be absolute or relative to the cruncher root (`src/dnadesign/cruncher`); relative paths must not include `..`.
- By default the catalog cache is shared across workspaces (`src/dnadesign/cruncher/.cruncher`); locks/run_index live in each workspace’s `.cruncher/`.

#### Workspace state (per workspace `.cruncher/`)

```
<workspace>/.cruncher/
locks/<config>.lock.json
run_index.json
```

- `locks/<config>.lock.json` pins TF names → exact cached artifacts + hashes.
- `run_index.json` tracks run folders for `cruncher runs ...` within that workspace.

#### Tooling caches

- Matplotlib writes its cache under `<catalog_root>/.mplcache/` unless `MPLCONFIGDIR` is set.
- Numba JIT cache defaults to `<repo>/src/dnadesign/cruncher/.cruncher/numba_cache` (or `<repo>/.cruncher/numba_cache`)
  unless `NUMBA_CACHE_DIR` is set.

#### Run outputs (`out_dir`, e.g. `outputs/`)

Each configured regulator set produces **separate** runs, grouped by stage. Run names include the TF slug (and a `setN_` prefix only when multiple regulator sets are configured):

- `outputs/parse/lexA-cpxR_20260101_143210_f3a9d2/`
- `outputs/sample/lexA-cpxR_20260101_143512_a91c0e/`
- `outputs/auto_opt/lexA-cpxR_20260101_143530_91acb1/` (auto-opt pilots)
- `outputs/logos/catalog/lexA-cpxR_20260101_143210_f3a9d2/` *(prefix `setN_` only when multiple regulator sets are configured)*

---

### Run artifacts

A typical **sample** run directory contains:

- `meta/` — manifests, config_used, live status snapshots
- `artifacts/` — sequences, trace (if enabled), elites exports
- `analysis/` — plots, tables, and analysis metadata (plus optional notebooks/archive)
- `live/` — streaming metrics (if enabled)
- `analysis/report.json` + `analysis/report.md` — analysis report outputs from `cruncher analyze`

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

@e-south
