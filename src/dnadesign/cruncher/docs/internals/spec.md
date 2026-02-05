## cruncher for developers


## Contents
- [cruncher for developers](#cruncher-for-developers)
- [Goals](#goals)
- [Architecture](#architecture)
- [Data model](#data-model)
- [Cache layout](#cache-layout)
- [Lockfiles](#lockfiles)
- [PWM creation strategy](#pwm-creation-strategy)
- [MCMC optimization spec](#mcmc-optimization-spec)
- [Outputs and reporting](#outputs-and-reporting)
- [CLI contract](#cli-contract)
- [Error handling](#error-handling)
- [Testing plan](#testing-plan)

### Goals

- Decoupled: core optimization is source-agnostic and runs offline.
- Assertive: explicit errors for missing inputs, ambiguous TFs, invalid matrices.
- Extendable: new sources and optimizers are adapters/registries.
- Reproducible: lockfiles + run manifests + deterministic seeds.
- Operational UX: clear CLI commands, deterministic cache, readable reports, crisp docs.
- No fallbacks: no implicit legacy modes, no silent fallbacks, no hidden network access.

---

### Architecture

#### Layers (ports & adapters)

- **core/** — PWM, scoring, evaluator, state, optimizers; no I/O.
- **ingest/** — source adapters (RegulonDB first), normalization, validation.
- **store/** — local catalog + lockfiles (project-local `.cruncher/` only).
- **analysis/** — analysis helpers, plot registry, per-PWM summaries.
- **artifacts/** — run layout, manifest + status helpers.
- **viz/** — plotting helpers + PWM logo rendering.
- **integrations/** — external tool wrappers (e.g., MEME Suite).
- **app/** — parse/sample/analyze orchestration + application services.
- **cli/** — Typer CLI, no business logic.

#### Registries

- **Source registry**: adapters are registered once, resolved by ID (including config-defined local sources).
- **Parser registry**: PWM parsers register via `io/parsers/` or `io.parsers.extra_modules`.
- **Optimizer registry**: kernels are registered once, resolved by name.

---

### Data model

#### MotifRecord (normalized)

- descriptor: source, motif_id, tf_name, organism, length, kind, tags
- tags.synonyms: optional semicolon-separated alias list for fuzzy discovery
- matrix: list[list[float]] (L x 4, probabilities)
- matrix_semantics: "probabilities" or "counts"
- provenance: source_url, version, license, citation, retrieved_at
- checksums: sha256_raw, sha256_norm

#### SiteInstance (normalized)

- source, site_id, motif_ref (source:motif_id)
- coordinate: 0-based half-open interval (optional)
- sequence: A/C/G/T string (optional; HT peaks may be coordinate-only)
- strand: "+" | "-" | None
- provenance: retrieval metadata

---

### Cache layout

```
<catalog_root>/
  catalog.json
  normalized/
    motifs/<source>/<motif_id>.json
    sites/<source>/<motif_id>.jsonl
  discoveries/
  .mplcache/
```

`catalog.json` is the single source of truth for “what we have in-house”. It tracks matrix availability, site counts, and provenance tags.
`catalog_root` can be absolute or relative to the cruncher root (`src/dnadesign/cruncher`); relative paths must not include `..`.
By default the catalog cache is shared across workspaces (`src/dnadesign/cruncher/.cruncher`).

Workspace state (per workspace `.cruncher/`):

```
<workspace>/.cruncher/
  locks/
    <config>.lock.json
  run_index.json
```

Tooling caches:

- Matplotlib caches in `<catalog_root>/.mplcache/` unless `MPLCONFIGDIR` is set.
- Numba JIT cache defaults to `<workspace>/.cruncher/numba_cache` unless `NUMBA_CACHE_DIR` is set.

---

### Lockfiles

Lockfiles pin TF names to exact source IDs and checksums. Lockfiles are **required** for:

- parse
- sample

If a TF cannot be uniquely resolved, **cruncher** errors immediately. Analyze operates on run artifacts and validates the lockfile recorded in the run manifest.

---

### PWM creation strategy

- Default: use cached matrices (`motif_store.pwm_source=matrix`).
- Optional: build PWM from cached sites (`motif_store.pwm_source=sites`).
- Site-derived PWMs use Biopython with configurable pseudocounts (`motif_store.pseudocounts`).
- `motif_store.site_kinds` can restrict which site sets are eligible (e.g., curated vs HT vs local).
- `motif_store.combine_sites=true` concatenates site sets for a TF before PWM creation (explicit opt‑in).
- When `combine_sites=true`, lockfiles hash the full set of site files used for that TF, so cache changes require re-locking.
- HT site sets with variable lengths require per‑TF/per‑dataset window lengths via `motif_store.site_window_lengths`.
- If optimization requires a shorter PWM, set `motif_store.pwm_window_lengths` to select the highest‑information window.
- Fail if fewer than `min_sites_for_pwm` binding sites are available (unless `allow_low_sites=true`).
- All PWMs are validated (shape Lx4, rows sum to 1, non-negative).
- De novo alignment/discovery is handled via MEME Suite (`cruncher discover motifs`) and stored as catalog matrices.
  Tool resolution uses `motif_discovery.tool_path` (resolved relative to config), `MEME_BIN`, or PATH, and discovery
  writes a `discover_manifest.json` with tool/version metadata per run.

---

### MCMC optimization spec

- Deterministic RNG via `sample.rng.seed` (and `sample.rng.deterministic=true` for stable pilot seeding).
- Burn-in storage is optional via `sample.output.trace.include_tune` (default: false, affects sequences.parquet only).
- Optimizer registry supports `pt` (parallel tempering).
- Each optimizer reports:
  - move tallies
  - acceptance ratios for B/M moves
  - PT swap acceptance rates
- Cooling (PT ladder) and soft-min schedules are independent.
- Optional adaptive controllers tune the PT ladder scale toward target swap acceptance.
- Move policies support slide/swap/insertion moves plus optional “worst TF” targeting and move scheduling.

---

### Outputs and reporting

Each run directory contains:

- `meta/config_used.yaml` — resolved config + PWM summaries
- `meta/run_manifest.json` — provenance, hashes, optimizer stats
- `meta/run_status.json` — live progress updates (written during parse and sampling)
- `artifacts/trace.nc` — canonical ArviZ trace
- `artifacts/sequences.parquet` — per-draw sequences + per-TF scores
- `artifacts/elites.parquet` — elite sequences (parquet)
- `artifacts/elites.json` — elite sequences (JSON, human-readable)
- `artifacts/elites.yaml` — elite metadata (YAML)
- `analysis/` — latest analysis (plots/tables/report in one directory)
- `analysis/summary.json` — analysis provenance and artifacts
- `analysis/manifest.json` — artifact inventory with generation reasons
- `analysis/analysis_used.yaml` — analysis settings used
- `analysis/plot_manifest.json` — plot registry and generated outputs
- `analysis/table_manifest.json` — table registry and generated outputs
- `analysis/auto_opt_pilots.parquet` — pilot scorecard (when auto-opt runs)
- `analysis/plot__auto_opt_tradeoffs.<plot_format>` — balance vs best-score tradeoffs (when auto-opt runs)
- `analysis/_archive/<analysis_id>/` — optional archived analyses (when enabled)
- `live/metrics.jsonl` — live sampling progress (when enabled)
- `analysis/report.json` + `analysis/report.md` — summary (from `cruncher analyze`)

`cruncher analyze` warns when required analysis artifacts are missing and still writes the report stubs.

---

### CLI contract

Most commands accept an explicit config `--config`. If omitted, **cruncher** resolves a config from `--workspace`/`CRUNCHER_WORKSPACE`, then from `config.yaml` in the current directory (or parent directories), and finally from discoverable workspaces. When multiple workspaces are discovered, Cruncher prompts in interactive shells.

Defaults + automation:

- `workspaces/.default_workspace` (or `<workspace_root>/.default_workspace`) can pin a workspace name or config path.
- `CRUNCHER_DEFAULT_WORKSPACE=<name>` selects a default when multiple workspaces exist.
- `CRUNCHER_NONINTERACTIVE=1` disables prompts and fails fast when ambiguous.
- `CRUNCHER_WORKSPACE_ROOTS=/path/a:/path/b` adds workspace search roots.

- `cruncher fetch motifs ...`
- `cruncher fetch sites ...`
- `cruncher fetch motifs --offline` (verify cached only)
- `cruncher fetch motifs --update` (force refresh)
- `cruncher lock <config>`
- `cruncher targets list <config>`
- `cruncher targets status <config>`
- `cruncher targets candidates <config>`
- `cruncher parse <config>`
- `cruncher sample <config>`
- `cruncher analyze --run <run_name|run_dir> <config>`
- `cruncher analyze --latest <config>`
- `cruncher analyze --summary <config>`
- `cruncher runs list <config> [--stage sample]`
- `cruncher runs show <config> <run_name>`
- `cruncher runs rebuild-index <config>`
- `cruncher notebook [--analysis-id <id>|--latest] <run_dir>`

Pairwise plots auto-pick a deterministic `tf_pair` when missing; explicit `analysis.tf_pair` overrides the selection.

Network access is explicit and opt-in. `cruncher fetch ...` and remote inventory commands (for example `cruncher sources summary --scope remote` or
`cruncher sources datasets`) contact sources; other commands operate on local
cache and run artifacts only.

Inspection:

- `cruncher catalog list <config> [--organism <name>]`
- `cruncher catalog search <config> <query> [--fuzzy]`
- `cruncher catalog resolve <config> <tf>`
- `cruncher catalog show <config> <source>:<motif_id>`
- `cruncher optimizers list`
- `cruncher config summary <config>`
- `cruncher cache verify <config>`
- `cruncher runs watch <config> <run_name>`

---

### Error handling

Errors are explicit and actionable:

- Missing lockfile → error (no implicit resolution)
- Lockfile pwm_source mismatch → error (re-run lock)
- Missing artifacts for analyze → error (no partial outputs)
- Invalid PWM / invalid sites → error
- Ambiguous TF resolution → error
- PWM-from-sites with low site count → error unless `allow_low_sites=true`

---

### Testing plan

Unit tests:

- PWM validation (shape, row sums)
- SequenceState validation (dtype/range)
- Scorer scale coverage
- Optimizer registry behavior
- Manifest serialization and hashing

Integration tests:

- fetch motifs/sites → catalog updates
- lock → parse/sample with catalog
- analysis report generation with run artifacts

End-to-end:

- LocalDir ingestion (offline, matrices only by default; MEME BLOCKS sites when extract_sites=true)
- sample + analyze with small draws

---

@e-south
