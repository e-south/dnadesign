## cruncher for developers

This document defines the end-to-end requirements and architecture for **cruncher**. It is intended as a build and review guide for engineers working on ingestion, optimization, and UX.

### Contents

1. [Goals](#goals)
2. [Architecture](#architecture)
3. [Registries](#registries)
4. [Data model](#data-model)
5. [Cache layout](#cache-layout)
6. [Lockfiles](#lockfiles)
7. [PWM creation strategy](#pwm-creation-strategy)
8. [MCMC optimization spec](#mcmc-optimization-spec)
9. [Outputs and reporting](#outputs-and-reporting)
10. [CLI contract](#cli-contract)
11. [Error handling](#error-handling)
12. [Testing plan](#testing-plan)

---

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
- **app/** — parse/sample/analyze/report orchestration + application services.
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
  run_index.json
  locks/
    <config>.lock.json
  normalized/
    motifs/<source>/<motif_id>.json
    sites/<source>/<motif_id>.jsonl
```

`catalog.json` is the single source of truth for “what we have in-house”. It tracks matrix availability, site counts, and provenance tags.
`catalog_root` must be workspace-relative (no absolute paths or `..` segments).

---

### Lockfiles

Lockfiles pin TF names to exact source IDs and checksums. Lockfiles are **required** for:

- parse
- sample

If a TF cannot be uniquely resolved, **cruncher** errors immediately. Analyze/report operate on run artifacts and validate the lockfile recorded in the run manifest.

---

### PWM creation strategy

- Default: use cached matrices (`motif_store.pwm_source=matrix`).
- Optional: build PWM from cached sites (`motif_store.pwm_source=sites`).
- Site-derived PWMs use Biopython with configurable pseudocounts (`motif_store.pseudocounts`).
- `motif_store.site_kinds` can restrict which site sets are eligible (e.g., curated vs HT vs local).
- `motif_store.combine_sites=true` concatenates site sets for a TF before PWM creation (explicit opt‑in).
- When `combine_sites=true`, lockfiles hash the full set of site files used for that TF, so cache changes require re-locking.
- HT site sets with variable lengths require per‑TF/per‑dataset window lengths via `motif_store.site_window_lengths`.
- If optimization requires a shorter PWM, set `motif_store.pwm_window_lengths` to trim to the highest‑information sub-window.
- Fail if fewer than `min_sites_for_pwm` binding sites are available (unless `allow_low_sites=true`).
- All PWMs are validated (shape Lx4, rows sum to 1, non-negative).
- De novo alignment/discovery is handled via MEME Suite (`cruncher discover motifs`) and stored as catalog matrices.
  Tool resolution uses `motif_discovery.tool_path` (resolved relative to config), `MEME_BIN`, or PATH, and discovery
  writes a `discover_manifest.json` with tool/version metadata per run.

---

### MCMC optimization spec

- Deterministic RNG via `sample.rng.seed` (and `sample.rng.deterministic=true` for stable pilot seeding).
- Burn-in storage is optional via `sample.output.trace.include_tune` (default: false).
- Optimizer registry supports `gibbs` and `pt` out of the box.
- Each optimizer reports:
  - move tallies
  - acceptance ratios for B/M moves
  - PT swap acceptance rates
- Cooling and soft-min schedules are independent; `optimizers.gibbs.apply_during` controls whether annealing happens during tune only or all sweeps.
- `optimizers.gibbs.schedule_scope` selects per‑chain vs global cooling schedules (global spans all chains and requires `apply_during=all`).
- Optional adaptive controllers tune Gibbs acceptance or PT ladder scale toward target bands.
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
- `analysis/` — latest analysis (plots/tables/notebooks)
- `analysis/meta/summary.json` — analysis provenance and artifacts
- `analysis/meta/analysis_used.yaml` — analysis settings used
- `analysis/meta/plot_manifest.json` — plot registry and generated outputs
- `analysis/meta/table_manifest.json` — table registry and generated outputs
- `analysis/tables/auto_opt_pilots.csv` — pilot scorecard (when auto-opt runs)
- `analysis/plots/auto_opt_tradeoffs.png` — balance vs best-score tradeoffs (when auto-opt runs)
- `analysis/_archive/<analysis_id>/` — optional archived analyses (when enabled)
- `live/metrics.jsonl` — live sampling progress (when enabled)
- `report/report.json` + `report/report.md` — summary (from `cruncher report`)

`cruncher report` **fails** if required artifacts are missing.

---

### CLI contract

Most commands accept an explicit config `--config`. If omitted, **cruncher** resolves a config from `--workspace`/`CRUNCHER_WORKSPACE`, then from `config.yaml` in the current directory (or parent directories), and finally from discoverable workspaces.

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
- `cruncher report <config> <run_name>`
- `cruncher runs list <config> [--stage sample]`
- `cruncher runs show <config> <run_name>`
- `cruncher runs rebuild-index <config>`
- `cruncher notebook [--analysis-id <id>|--latest] <run_dir>`

Pairwise plots only run when `analysis.tf_pair` is set; there is no implicit pairwise sweep.

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
- Missing artifacts for analyze/report → error
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
- report generation with run artifacts

End-to-end:

- LocalDir ingestion (offline, matrices only by default; MEME BLOCKS sites when extract_sites=true)
- sample + analyze + report with small draws

---

@e-south
