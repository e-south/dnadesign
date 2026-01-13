## cruncher architecture

**cruncher** is structured so that data access, sequence optimization logic, and the CLI can evolve independently.


### Contents

1. [Run lifecycle](#run-lifecycle)
2. [Layers and responsibilities](#layers-and-responsibilities)
3. [On-disk layout](#on-disk-layout)
4. [Run artifacts](#run-artifacts)
5. [Reproducibility boundaries](#reproducibility-boundaries)
6. [Extensibility points](#extensibility-points)

---

### Run lifecycle

1. **fetch**
   - populates `normalized/` (motifs and/or sites)
   - updates `catalog.json`

2. **lock**
   - resolves TF names in `regulator_sets` to exact cached candidates
   - writes `.cruncher/locks/<config>.lock.json`
   - locking is what makes later stages reproducible

3. **parse** (optional)
   - validates cached motifs (and/or site-derived PWMs)
   - renders logos and basic QC artifacts
   - records provenance into a run manifest

4. **sample**
   - loads the locked PWMs
   - runs MCMC optimization
   - writes sequences + (optionally) traces + manifests

5. **analyze**
   - generates plots/tables from sample artifacts
   - never calls sources or the network

6. **report**
   - produces `report/report.json` and `report/report.md` from run artifacts
   - fails fast if required artifacts are missing

---

### Layers and responsibilities

The core contract is:

- **Network access is explicit** (fetch commands and optional remote inventory commands like
  `cruncher sources summary --scope remote` and `cruncher sources datasets`).
- The **store** is the only persistence layer (project-local; no global state).
- The **core** (PWM scoring + optimizers) is pure compute and does no I/O.
- **Analyze/report** consume run artifacts only and can run fully offline.

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

#### `workflows/` (orchestration)
- fetch / lock / parse / sample / analyze / report coordination
- translates CLI intent + config into concrete runs and artifacts

#### `cli/` (UX only)
- Typer commands
- argument parsing, output formatting
- delegates work to services/workflows (no business logic)

---

### On-disk layout

**cruncher** uses **project-local state** (relative to the config file). No global cache.

Recommended workspace layout:

```
<workspace>/
config.yaml
.cruncher/
runs/
```

In this repo, the bundled demo workspaces live at:

- `src/dnadesign/cruncher/workspaces/demo_basics_two_tf/`
- `src/dnadesign/cruncher/workspaces/demo_campaigns_multi_tf/`

#### Catalog root (`catalog_root`, default: `.cruncher/`)

```
<catalog_root>/
catalog.json
run_index.json
locks/ <config>.lock.json
normalized/
motifs/<source>/<motif_id>.json
sites/<source>/<motif_id>.jsonl
genomes/              # if genome hydration is enabled
```

- `catalog.json` is the “what do we have cached?” index.
- `locks/<config>.lock.json` pins TF names → exact cached artifacts + hashes.
- `run_index.json` tracks run folders for `cruncher runs ...`.
- `catalog_root` must be workspace-relative (no absolute paths or `..` segments).

#### Run outputs (`out_dir`, e.g. `runs/`)

Each configured regulator set produces **separate** runs, grouped by stage and regulator set:

- `runs/parse/set1_lexA-cpxR/set1_lexA-cpxR_20260101_143210_f3a9d2/`
- `runs/sample/set1_lexA-cpxR/set1_lexA-cpxR_20260101_143512_a91c0e/`
- `runs/pilot/set1_lexA-cpxR/set1_lexA-cpxR_20260101_143530_91acb1/` (auto-opt pilots)
- `runs/logos/parse/set1_lexA-cpxR_20260101_143210_f3a9d2/`
- `runs/logos/catalog/set1_lexA-cpxR_20260101_143210_f3a9d2/`

---

### Run artifacts

A typical **sample** run directory contains:

- `meta/` — metadata + provenance
  - `meta/config_used.yaml` — resolved runtime config + PWM summaries
  - `meta/run_manifest.json` — provenance, resolved motifs, optimizer stats, hashes
  - `meta/run_status.json` — live progress snapshots during parse/sample
- `artifacts/` — generated outputs
  - `artifacts/sequences.parquet` — per-draw sequences + per-TF scores *(required for analyze/report)*
  - `artifacts/trace.nc` — ArviZ-compatible trace *(required for trace-based plots and some report metrics)*
  - `artifacts/elites.parquet` / `elites.json` / `elites.yaml` — elite sequences and exports
- `analysis/` — latest analysis artifacts
  - `analysis/meta/summary.json` — analysis summary + manifest links
  - `analysis/meta/analysis_used.yaml` — analysis settings (resolved)
  - `analysis/meta/plot_manifest.json` — plot registry output
  - `analysis/meta/table_manifest.json` — table registry output
  - `analysis/plots/` — plots (PNG/PDF)
  - `analysis/tables/` — CSV/JSON tables
  - `analysis/notebooks/` — optional notebooks
- `analysis/_archive/<analysis_id>/` — optional archived analyses (when enabled)
- `live/metrics.jsonl` — live sampling progress snapshots (when enabled)
- `report/` — generated summaries (from `cruncher report`)

---

### Reproducibility boundaries

- **Lockfiles are mandatory** for `parse` and `sample`.
- If you change inputs that affect TF resolution (e.g., PWM source, site filters, dataset selection),
  **re-lock** so the lockfile hash set matches reality.
- `analyze` and `report` validate the lockfile recorded in the run manifest.

---

### Extensibility points

- **Sources:** add a new adapter under `ingest/adapters/` and register it in the source registry.
- **Local sources:** configure `ingest.local_sources` for local motif directories (no new code required).
- **Parsers:** add a parser under `io/parsers/` or register via `io.parsers.extra_modules`.
- **Optimizers:** add a new kernel and register it in `core/optimizers/registry.py`.
- **Analysis plots:** add a plot implementation and register it in the analysis plot registry.

---

@e-south
