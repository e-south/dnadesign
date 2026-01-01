# Cruncher Refactor Spec (Developer-Ready)

This document defines the end-to-end requirements and architecture for Cruncher after the refactor.
It is intended as a build and review guide for engineers working on ingestion, optimization, and UX.

## 1) Goals (non-negotiable)

- Decoupled: core optimization is source-agnostic and runs offline.
- Assertive: explicit errors for missing inputs, ambiguous TFs, invalid matrices.
- Extendable: new sources and optimizers are adapters/registries, not rewrites.
- Reproducible: lockfiles + run manifests + deterministic seeds.
- Operational UX: clear CLI commands, deterministic cache, readable reports, crisp docs.
- No fallbacks: no implicit legacy modes, no silent fallbacks, no hidden network access.

## 2) Non-goals (v1)

- Full derivation of motifs from raw sequencing reads.
- Genome coordinate liftover or orthology mapping.
- Hosted service or server-backed catalog.

## 3) Architecture

### Layers (ports & adapters)

- **core/** — PWM, scoring, evaluator, state, optimizers; no I/O.
- **ingest/** — source adapters (RegulonDB first), normalization, validation.
- **store/** — local catalog + lockfiles (project-local `.cruncher/` only).
- **workflows/** — parse/sample/analyze/report orchestration.
- **cli/** — Typer CLI, no business logic.

### Registries

- **Source registry**: adapters are registered once, resolved by ID.
- **Optimizer registry**: kernels are registered once, resolved by name.

## 4) Data model (canonical)

### MotifRecord (normalized)

- descriptor: source, motif_id, tf_name, organism, length, kind, tags
- tags.synonyms: optional semicolon-separated alias list for fuzzy discovery
- matrix: list[list[float]] (L x 4, probabilities)
- matrix_semantics: "probabilities" (v1 only)
- provenance: source_url, version, license, citation, retrieved_at
- checksums: sha256_raw, sha256_norm

### SiteInstance (normalized)

- source, site_id, motif_ref (source:motif_id)
- coordinate: 0-based half-open interval (optional)
- sequence: A/C/G/T string (optional; HT peaks may be coordinate-only)
- strand: "+" | "-" | None
- provenance: retrieval metadata

## 5) Cache layout (project-local)

```
.cruncher/
  catalog.json
  run_index.json
  locks/
    <config>.lock.json
  normalized/
    motifs/<source>/<motif_id>.json
    sites/<source>/<motif_id>.jsonl
```

`catalog.json` is the single source of truth for “what we have in-house”.
It tracks matrix availability, site counts, and provenance tags.

## 6) Lockfiles

Lockfiles pin TF names to exact source IDs and checksums. Lockfiles are **required** for:

- parse
- sample
- analyze
- report

If a TF cannot be uniquely resolved, Cruncher errors immediately.

## 7) PWM creation strategy

- Default: use cached matrices (`motif_store.pwm_source=matrix`).
- Optional: build PWM from cached sites (`motif_store.pwm_source=sites`).
- `motif_store.site_kinds` can restrict which site sets are eligible (e.g., curated vs HT).
- `motif_store.combine_sites=true` concatenates site sets for a TF before PWM creation (explicit opt‑in).
- HT site sets with variable lengths require per‑TF/per‑dataset window lengths via `motif_store.site_window_lengths`.
- Fail if fewer than `min_sites_for_pwm` binding sites are available (unless `allow_low_sites=true`).
- All PWMs are validated (shape Lx4, rows sum to 1, non-negative).

## 8) MCMC optimization spec

- Deterministic RNG via `sample.seed`.
- Burn-in storage is optional via `sample.record_tune` (default: false).
- Optimizer registry supports `gibbs` and `pt` out of the box.
- Each optimizer reports:
  - move tallies
  - acceptance ratios for B/M moves
  - PT swap acceptance rates

## 9) Outputs & reporting

Each run directory contains:

- `config_used.yaml` — resolved config + PWM summaries
- `trace.nc` — canonical ArviZ trace
- `sequences.parquet` — per-draw sequences + per-TF scores
- `cruncher_elites_*/<name>.parquet` — elite sequences (parquet)
- `cruncher_elites_*/<name>.json` — elite sequences (JSON, human-readable)
- `run_manifest.json` — provenance, hashes, optimizer stats
- `run_status.json` — live progress updates (written during parse and sampling)
- `report.json` + `report.md` — summary (from `cruncher report`)

`cruncher report` **fails** if required artifacts are missing.

## 10) CLI contract

Core lifecycle:

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
- `cruncher analyze <config>`
- `cruncher report <config> <batch_name>`
- `cruncher runs list <config> [--stage sample]`
- `cruncher runs show <config> <run_name>`
- `cruncher runs rebuild-index <config>`

No command performs hidden network access. Fetching is explicit.

Inspection:

- `cruncher catalog list <config> [--organism <name>]`
- `cruncher catalog search <config> <query> [--fuzzy]`
- `cruncher catalog resolve <config> <tf>`
- `cruncher catalog show <config> <source>:<motif_id>`
- `cruncher optimizers list`
- `cruncher config summary <config>`
- `cruncher cache verify <config>`
- `cruncher runs watch <config> <run_name>`

## 11) Error handling (assertive)

Errors are explicit and actionable:

- Missing lockfile → error (no implicit resolution)
- Lockfile pwm_source mismatch → error (re-run lock)
- Missing artifacts for analyze/report → error
- Invalid PWM / invalid sites → error
- Ambiguous TF resolution → error
- PWM-from-sites with low site count → warning (not silent)

## 12) Testing plan (minimum)

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

- LocalDir ingestion (offline)
- sample + analyze + report with small draws
