## Cruncher architecture

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-28


**Last updated by:** cruncher-maintainers on 2026-02-28

### Contents
- [Cruncher architecture](#cruncher-architecture)
- [Run lifecycle](#run-lifecycle)
- [Layers and responsibilities](#layers-and-responsibilities)
- [On-disk layout](#on-disk-layout)
- [Run artifacts](#run-artifacts)
- [Study artifacts](#study-artifacts)
- [Portfolio artifacts](#portfolio-artifacts)
- [Reproducibility boundaries](#reproducibility-boundaries)
- [Extensibility points](#extensibility-points)
- [Related docs](#related-docs)

This doc describes the Cruncher run lifecycle, module boundaries, and on-disk artifacts.

#### Run lifecycle

1. **fetch** -> cache motifs/sites and update `catalog.json`
2. **lock** -> resolve TFs to exact cached artifacts (`<workspace>/.cruncher/locks/<config>.lock.json`)
3. **parse** *(optional)* -> validate locked PWMs and refresh the parse cache in workspace state (no logo rendering)
4. **sample** -> run MCMC and write sequences/trace + manifests
5. **analyze** -> curated `plots/*` and `tables/table__*` artifacts + report from sample artifacts (offline, written into the run directory)
6. **export** -> sequence-centric contract tables for wrappers/operators (`cruncher export sequences`)

---

#### Layers and responsibilities

Core contract:

- **Network access is explicit** (fetch and remote inventory).
- The **store** is the only persistence layer (project-local).
- The **core** (PWM scoring + optimizers) is pure compute (no I/O).
- **Analyze** reads run artifacts only and can run offline.

#### `core/` (pure compute)
- PWM representation and validation
- scoring / evaluator logic
- sequence state and move operators
- optimizer kernels (gibbs annealing)
- No I/O (no filesystem, no network)

#### `ingest/` (ports/adapters)
- source adapters (RegulonDB first)
- normalization into standard records (motifs + sites)
- optional hydration (coordinates -> sequences) via genome providers

#### `store/` (local persistence)
- the on-disk catalog (what's cached)
- lockfiles (what's pinned)
- run index (what runs exist)

#### `analysis/` (analysis + diagnostics)
- plot registry, per-PWM summaries, and analysis helpers
- plot implementations live under `analysis/plots/`
- baserender-backed elites showcase lives in `analysis/plots/elites_showcase.py`
- trajectory score-space elite mapping/sampling helpers live in `analysis/plots/trajectory_score_space.py`
- trajectory score-space panel rendering helpers live in `analysis/plots/trajectory_score_space_panel.py`
- trajectory score-space plot orchestration lives in `analysis/plots/trajectory_score_space_plot.py`
- chain-trajectory video orchestration lives in `analysis/trajectory_video.py`
- trajectory frame/timeline selection helpers live in `analysis/trajectory_video_timeline.py`
- baserender video contract assembly lives in `analysis/trajectory_video_contract.py`

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
- study coordination (`study run|summarize|show`)
- portfolio coordination (`portfolio run|show`)
- translates CLI intent + config into concrete runs and artifacts

#### `cli/` (UX only)
- Typer commands
- argument parsing, output formatting
- delegates work to app modules (no business logic)

#### Baserender integration boundary

Cruncher integrates with baserender through the **public package root only**:

- Allowed: `from dnadesign.baserender import ...`
- Disallowed: `dnadesign.baserender.src.*` deep imports

Current Cruncher handoff for `elites_showcase.*` and `chain_trajectory_video.mp4`:

1. Cruncher resolves run data into rendering primitives:
   - sequence per elite
   - best-window spans/strand per TF
   - locked motif matrices for each TF
2. Cruncher hands baserender only the minimal plotting contract:
   - record-shaped rows (`id`, `sequence`, `features`, `effects`, `display`) and motif primitives
   - or equivalent in-memory `Record` objects through baserender public APIs
3. Baserender validates contracts, performs layout/rendering, and emits assets.

For `chain_trajectory_video.mp4`, Cruncher first resolves selected-chain trajectory rows and sampled frame indices, then writes temporary record rows and passes a strict sequence-rows video job contract to baserender.

The showcase/video renderers do not require overlap tables; overlap metrics remain separate analysis artifacts.

This keeps responsibilities decoupled:
- Cruncher owns analysis semantics and motif provenance.
- Baserender owns rendering contracts, geometry, and output encoding.
- Both sides fail fast on schema/contract violations.

---

#### On-disk layout

**cruncher** uses **project-local state** (relative to the workspace root resolved from config). Data artifacts live in the workspace;
tooling caches stay within the repo/workspace unless you override their environment variables.

Recommended workspace layout:

```
<workspace>/
configs/
  config.yaml
  studies/               # optional study specs
  portfolios/            # optional portfolio specs
.cruncher/
outputs/
```

In this repo, the bundled demo workspaces live at:

- `src/dnadesign/cruncher/workspaces/demo_pairwise/`
- `src/dnadesign/cruncher/workspaces/demo_multitf/`
- `src/dnadesign/cruncher/workspaces/project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs/`

#### Catalog root (`catalog.root`, default: `.cruncher/`)

```
<catalog.root>/
catalog.json
normalized/
motifs/<source>/<motif_id>.json
sites/<source>/<motif_id>.jsonl
genomes/              # if genome hydration is enabled
discoveries/          # MEME/STREME discovery runs
```

- `catalog.json` is the "what do we have cached?" index.
- `catalog.root` can be absolute or relative to the workspace root; relative paths must not include `..`.
- By default the catalog cache is workspace-local (`<workspace>/.cruncher`); locks/run_index also live in each workspace's `.cruncher/`.

#### Workspace state (per workspace `.cruncher/`)

```
<workspace>/.cruncher/
locks/<config>.lock.json
run_index.json
parse/inputs/{lockfile.json,parse_manifest.json,pwm_summary.json}
```

- `locks/<config>.lock.json` pins TF names -> exact cached artifacts + hashes.
- `run_index.json` tracks run folders for `cruncher runs ...` within that workspace.
- `parse/` stores parse-stage validation artifacts outside user-facing sample outputs.

#### Tooling caches

- Matplotlib writes its cache under `.cache/matplotlib/cruncher` unless `MPLCONFIGDIR` is set.
- Numba JIT cache defaults to `<workspace>/.cruncher/numba_cache` unless `NUMBA_CACHE_DIR` is set.

#### Run outputs (`out_dir`, e.g. `outputs/`)

Each regulator set gets one standard run directory:

- single regulator set: `<workspace>/<out_dir>/`
- multiple regulator sets: `<workspace>/<out_dir>/setN_<tf-slug>/`

Within each run directory, Cruncher uses a stable, stage-agnostic subdirectory layout:

```
<run_dir>/
  meta/
  provenance/
  optimize/
  analysis/
  plots/
  export/
```

---

#### Run artifacts

A typical **sample** run directory contains:

- `meta/run_manifest.json`, `meta/run_status.json`, `meta/config_used.yaml` - run metadata + status
- `provenance/lockfile.json` - pinned input snapshot (reproducibility boundary)
- `optimize/tables/sequences.parquet`, `optimize/tables/elites*`, `optimize/tables/random_baseline*` - sampling tables (`random_baseline*` defaults on with `sample.output.save_random_baseline=true`, `sample.output.random_baseline_n=10000`)
- `optimize/state/trace.nc`, `optimize/state/metrics.jsonl`, `optimize/state/elites.{json,yaml}` - sampling metadata
- `analysis/reports/summary.json` - standard analysis summary
- `analysis/reports/report.json` + `analysis/reports/report.md` - analysis report outputs from `cruncher analyze`
- `analysis/manifests/plot_manifest.json` + `analysis/manifests/table_manifest.json` + `analysis/manifests/manifest.json` - analysis inventories
- `export/table__elites.csv` + `export/table__*.{parquet|csv}` + `export/export_manifest.json` - sequence-export tables from `cruncher export sequences`
- `plots/*` - curated analysis plots and catalog logo renders
- `plots/chain_trajectory_video.mp4` - optional trajectory-video artifact (`analysis.trajectory_video.enabled=true`)
- `analysis/tables/table__*` - curated table outputs

---

#### Study artifacts

Study runs are aggregate sweep workflows that keep deterministic workspace config separate from sweep intent.

Study specs live under:

```
<workspace>/configs/studies/<name>.study.yaml
```

Study outputs live under:

```
<workspace>/outputs/studies/<study.name>/<study_id>/
  study/
  trials/
  tables/
  manifests/
```

Key points:

- `study_id` is deterministic from frozen spec + base config hash + target descriptor.
- Study specs support both explicit trial lists and cartesian grid expansion (`trial_grids`).
- Trial outputs are nested under `trials/<trial_id>/seed_<seed>/`.
- `tables/` stores aggregate sweep tables (`table__trial_metrics*`, optional `table__mmr_tradeoff_agg`).
- Aggregate study plots are workspace-flat under `outputs/plots/` as namespaced files
  (`study__<study_name>__<study_id>__plot__*`).
- Study trial sampling does **not** write to workspace `run_index.json`.
- Replay sweeps reuse saved run artifacts instead of re-sampling where possible (MMR selection replay).

---

#### Portfolio artifacts

Portfolio runs aggregate selected source runs across workspaces into one handoff package.

Portfolio specs live under:

```
<portfolio_workspace>/configs/<name>.portfolio.yaml
```

Portfolio outputs live under:

```
<portfolio_workspace>/outputs/<portfolio_name>/<portfolio_id>/
  meta/
  tables/
  plots/
```

Key points:

- `portfolio_id` is deterministic from the frozen Portfolio spec payload.
- Source run selection is explicit in spec (`workspace`, `run_dir` per source).
- Source elite count is contract-driven from source run manifest `top_k` and `elites.parquet`.
- Portfolio plots are run-scoped under
  `<portfolio_run_dir>/plots/plot__*`.
- No implicit latest-run fallback is used during portfolio aggregation.
- Handoff tables are source-provenance-first (`table__handoff_windows_long`, `table__handoff_elites_summary`, `table__source_summary`).

---

#### Reproducibility boundaries

- **Lockfiles are mandatory** for `parse` and `sample`.
- If you change inputs that affect TF resolution (e.g., PWM source, site filters, dataset selection),
  **re-lock** so the lockfile hash set matches reality.
- `analyze` validates the lockfile recorded in the run manifest.

---

#### Extensibility points

- **Sources:** add a new adapter under `ingest/adapters/` and register it in the source registry.
- **Local sources:** configure `ingest.local_sources` for local motif directories (no new code required).
- **Parsers:** add a parser under `io/parsers/` or register via `io.parsers.extra_modules`.
- **Optimizers:** add a new kernel and register it in `core/optimizers/registry.py`.
- **Analysis plots:** add a plot implementation and register it in the analysis plot registry.

---

#### Related docs

- [Intent + lifecycle](../guides/intent_and_lifecycle.md)
- [Portfolio aggregation](../guides/portfolio_aggregation.md)
- [Config reference](config.md)

@e-south
