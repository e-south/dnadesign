# DenseGen Overhaul Roadmap (Breaking Changes, No Fallbacks)

This roadmap is the persistent source of truth for refactoring DenseGen into a
clean, decoupled, testable pipeline with strict, explicit behavior. There is
**no backward compatibility** and **no silent fallback behavior**. Any relaxation
is explicit, policy-driven, and recorded in outputs.

## Principles (Non‑Negotiable)

- **No fallbacks**: If a condition cannot be satisfied, fail fast with a clear
  error and actionable diagnostics.
- **Explicit policies**: Any adaptive behavior is opt‑in and recorded.
- **Decoupled architecture**: Components are swappable via clear interfaces.
- **Assertive programming**: Validate inputs and invariants early and often.
- **Determinism**: Seeded, reproducible runs by default.
- **Canonical identity**: A single ID scheme (USR `compute_id`).
- **Semantic alignment**: CLI, docs, config schema, and outputs match exactly.

## Target Architecture (Modules & Boundaries)

Adopt an explicit package layout under `dnadesign/densegen/src/` with clear boundaries:

- `config/` — Pydantic schema (versioned), strict validation (`extra="forbid"`).
- `core/` — Orchestration and domain logic (pipeline, sampler, metadata, postprocess, canonical).
- `adapters/sources/` — Data sources (binding-site tables, PWM MEME/JASPAR/CSV, sequence libraries, USR).
- `adapters/optimizer/` — Dense‑arrays adapter(s), constraint normalization.
- `adapters/outputs/` — Output sinks (USR, Parquet) with a shared schema writer.
- `viz/` — Schema‑driven plots; strict options.
- `utils/` — Logging + shared helpers.
- `cli.py` — Thin Typer CLI mapping config → pipeline.

## Canonical ID Scheme

Use USR’s `compute_id(bio_type, sequence_norm)` as the **only** record ID.
Parquet and USR outputs must share the same `id`.

## Output Schema (Required Fields)

All outputs must include:

- `id` (USR compute_id)
- `sequence`, `bio_type`, `alphabet`, `source`
- `densegen__schema_version`
- `densegen__created_at` (ISO8601, UTC)
- `densegen__length`
- `densegen__random_seed`
- `densegen__policy_gc_fill`
- `densegen__policy_sampling`
- `densegen__policy_solver`

Any derived fields must be documented in one place (schema reference) and used
consistently in plots and docs.

## Policies (No Fallbacks)

### GC Fill Policy
- **strict**: enforce `gc_min/max`; if infeasible, fail with a specific error.
- **adaptive** (opt‑in): compute feasible GC window given gap length, record
  `densegen__gap_fill_relaxed=true` and final GC bounds. Still fails if
  configured target is impossible and adaptive is disabled.

### Sampling Policy
- Coverage and uniqueness rules are strict by default.
- Any relaxation (e.g., allow replacement, allow duplicates) must be explicit
  in config and recorded in outputs.

### Solver Policy
- Strict solver selection: probe exactly the requested backend and fail if
  unavailable.

## Determinism

- A single seed controls Python + NumPy RNGs.
- Seed is recorded in output metadata.

## Global Runtime Guards

- Add `max_total_resamples`, `max_seconds_per_plan`, and `max_failed_solutions`.
- Any guard breach fails the run with diagnostics.

## Doc & CLI Alignment

- CLI help text must match schema terms exactly.
- `dense ls-plots` (or equivalent) must mirror the available plot registry.
- Example configs must be validated by the schema.

## Work Breakdown (Living Checklist)

### Phase 7 — Input Ergonomics & PWM Integration (current focus)
- [x] Define a canonical BindingSiteRecord schema (`regulator`, `sequence`, optional `site_id`, `source`).
- [x] Add a binding-site table input (CSV/Parquet) that maps directly to BindingSiteRecord.
- [x] Add PWM input: `pwm_meme` with strict validation.
- [x] Add PWM inputs: `pwm_jaspar`, `pwm_matrix_csv`.
- [x] Implement PWM sampling policies (`consensus | stochastic | background`) with explicit thresholds.
- [x] Add regulator constraints (per-plan `required_regulators` enforced in pipeline).
- [x] Extend regulator constraints (`min_required_regulators`, `min_count_by_regulator`).
- [x] Surface dense-arrays strand mode (`solver.strands`).
- [x] Add sampling pool strategy (`full | subsample | iterative_subsample`) with stop criteria.
- [x] Update docs + examples to cover binding sites vs PWM inputs and sampling flows.
- [x] Add tests: PWM parsing, sampling strategy boundaries, required regulator constraints.
- [ ] Add optional USR parity test (when USR installed) for canonical ID validation.

### Completed Phases (summary)
- **Phase 0–1**: Strict config schema + policy enforcement + input validation tests.
- **Phase 2–2b**: Pipeline refactor, IO separation, DI factories, optimizer adapter, postprocess module.
- **Phase 3–3b**: Canonical IDs + output schema, metadata registry, native Parquet lists/structs, ID index.
- **Phase 4**: Plotting strictly schema‑driven.
- **Phase 5**: Docs/CLI alignment + demo flow.
- **Phase 6**: UX/spec alignment (schema_version required, seaborn strictness, etc.).

## Decision Log

- **2026‑01‑13**: Breaking overhaul; no backward compatibility.
- **2026‑01‑15**: Package layout moved to `dnadesign/densegen/src/{core,adapters,config,viz}` (nested `src`).
- **2026‑01‑13**: Roadmap lives in `src/dnadesign/densegen/docs/`.
- **2026‑01‑13**: Canonical ID = USR `compute_id` (single scheme).
- **2026‑01‑13**: No fallbacks; any adaptation is explicit and recorded.
- **2026‑01‑13**: Replace `output.kind` with `output.targets` (explicit sink list).
- **2026‑01‑13**: Parquet output uses a dataset directory (`part-*.parquet`).
- **2026‑01‑14**: JSONL outputs removed; Parquet is the canonical non‑USR format.
- **2026‑01‑14**: Output schema centralized in `output.schema` (bio_type/alphabet shared by all sinks).
- **2026‑01‑14**: Config requires `densegen.schema_version` (currently `2.1`).
- **2026‑01‑14**: Binding-site inputs replace csv_tfbs/csv_sequences (no thin aliases).
- **2026‑01‑14**: PWM MEME inputs + sampling policies added.
- **2026‑01‑14**: PWM JASPAR + matrix CSV inputs added.
- **2026‑01‑14**: Sampling pool strategies (`full|subsample|iterative_subsample`) added.
- **2026‑01‑14**: Per-plan required_regulators enforced in pipeline.
- **2026‑01‑15**: Regulator constraints wired into dense-arrays (required/min counts + k-of-n).
- **2026‑01‑15**: Solver strand mode (`solver.strands`) added to config + metadata.
- **2026‑01‑15**: Run-scoped I/O via `densegen.run` (outputs/logs/plots confined to run root).
- **2026‑01‑14**: Promoter motifs validated strictly (non-empty A/C/G/T).
- **2026‑01‑14**: Parquet alignment loads dataset IDs even when dedup is disabled.
- **2026‑01‑14**: Plot styling is strict when `seaborn_style: true`.
- **2026‑01‑14**: Pipeline uses injected factories and optimizer adapter; gap fill moved to `postprocess/`.
- **2026‑01‑14**: `solver.backend` optional when `solver.strategy: approximate`.
- **2026‑01‑14**: Metadata schema registry enforced for output fields.
- **2026‑01‑14**: Parquet metadata uses native list/struct columns (no JSON encoding).
- **2026‑01‑14**: `logging.log_dir` required (explicit logs; no implicit `logs/`).
