# DenseGen Developer Specification (living)

This document defines **implementation contracts** and invariants for DenseGen. It is updated over time
as the architecture evolves.

## Invariants (assertive by default)

- **No fallbacks:** unknown config keys, invalid values, or ambiguous plan definitions are hard errors.
- **Explicit policies:** sampling, solver, and GC-fill behavior are always recorded in metadata.
- **Canonical IDs:** Parquet and USR must use the same deterministic ID scheme.
- **Output alignment:** when `output.targets` has multiple sinks, all outputs must be in sync; mismatches are errors.
- **Shared record builder:** a single OutputRecord constructor defines core fields for all sinks.
- **Run-scoped I/O:** configs must live inside `densegen.run.root`; outputs/logs/plots must resolve inside the run root.
- **USR optionality:** USR input/output is an optional integration; Parquet-only usage must not import USR code.
- **Determinism:** RNG seeds are explicit and passed into sampling + gap fill.
- **Relative paths:** inputs resolve against the config file directory; outputs/logs/plots resolve within `densegen.run.root`.
- **Explicit logs:** `logging.log_dir` is required; no implicit `logs/` fallback.
- **Parquet schema:** existing Parquet datasets must match the current schema; mismatches fail fast.
- **ID index:** Parquet outputs maintain a local `_densegen_ids.sqlite` index for fast dedup/alignment.
- **Valid motifs:** TFBS and sequence inputs must be A/C/G/T only.
- **Side biases:** `side_biases` motifs must be A/C/G/T, disjoint left/right, and present in the sampled library.
- **Solver backend:** `solver.backend` is required unless `solver.strategy: approximate`.

## Input model (implemented)

DenseGen uses a **single binding-site record model** regardless of input source:

- **BindingSiteRecord** fields (canonical):
  - `regulator` (TF name; non-empty string)
  - `sequence` (A/C/G/T only; normalized to uppercase)
  - `site_id` (optional stable identifier from source)
  - `source` (optional source identifier/path)
- `binding_sites` and PWM inputs (`pwm_meme`, `pwm_jaspar`, `pwm_matrix_csv`) map to this model.

### PWM inputs (implemented; MEME + JASPAR + CSV)

PWM inputs produce BindingSiteRecords via **explicit sampling policies**:

- Supported formats:
  - `pwm_meme` (MEME motif files; multi-motif).
  - `pwm_jaspar` (JASPAR PFM files; multi-motif).
  - `pwm_matrix_csv` (CSV matrix with A/C/G/T columns; single motif).
- Sampling policies are explicit in config and recorded in output metadata:
  - `strategy`: `consensus | stochastic | background`
  - `score_threshold` **or** `score_percentile` (exactly one required)
  - `n_sites` (required)
  - `oversample_factor` (required)
- All sampled records are tagged with `densegen__input_mode: pwm_sampled` and `densegen__input_pwm_ids`.

## Sampling flows (implemented)

To keep solver time bounded while preserving fidelity:

- `sampling.pool_strategy`: `full | subsample | iterative_subsample`
  - `full`: use all binding sites (max fidelity, max solver time)
  - `subsample`: single stratified subsample (fast, lower coverage)
  - `iterative_subsample`: resample libraries across attempts with explicit stop criteria
- `iterative_subsample` configuration:
  - `library_size`, `iterative_max_libraries`, `iterative_min_new_solutions`
  - Stop when quota met, libraries exhausted, or stall criteria reached (no fallback).
- Sampling is stratified by regulator when regulators are present.

## Regulator constraints (implemented)

- DenseGen passes regulator mappings into **dense-arrays** via `add_regulator_constraints`.
- Per-plan constraints:
  - `required_regulators` enforces **at least one site per required regulator**.
  - `min_required_regulators` enforces **k-of-n** regulators across the library.
  - `min_count_by_regulator` enforces per-regulator minimum counts.
- Solver strategies (`iterate|diverse|optimal`) enforce constraints at the solver level.
- `approximate` (heuristic) does not use solver constraints; the pipeline validates constraints
  to preserve behavior and fail fast on infeasible results.
- If infeasible, DenseGen fails fast with a clear diagnostic (no fallback).

## Configuration contracts

- `densegen` is required at the YAML root.
- `densegen.schema_version` is required and must be a supported version (currently `2.1`).
- `densegen.run` is required with `id` and `root` (run-scoped I/O).
- `generation.plan` must be non-empty and use **either all quotas or all fractions**.
- `output.schema` defines `bio_type` and `alphabet` for all outputs.
- `output.targets` controls required sub-blocks (`output.usr`, `output.parquet`).
- If `output.targets` has multiple sinks, then `plots.source` must be set to one of them.
- `solver.strategy` controls solution ordering (`iterate | diverse | optimal | approximate`).
- `solver.backend` is optional when `solver.strategy: approximate` and ignored by dense-arrays.
- `solver.strands` controls dense-arrays strand mode (`single | double`).
- `generation.plan.min_required_regulators` enforces k‑of‑n regulator coverage.
- `generation.plan.min_count_by_regulator` enforces per‑regulator minimum counts.

## GC gap fill policy

- `gap_fill.mode` is one of `off | strict | adaptive`.
- If `mode: off` and packed motifs are short, **fail fast**.
- Infeasible GC targets can occur for very short gaps (e.g., 1 nt cannot satisfy mid‑range targets).
  - `strict` must raise an error.
  - `adaptive` relaxes the GC window and records the relaxation in metadata.

## Canonical ID scheme

- IDs are computed using USR’s canonical algorithm:
  - `normalize_sequence(sequence, bio_type, alphabet)`
  - `compute_id(bio_type, normalized_sequence)`
- DenseGen implements this algorithm locally to avoid a hard dependency on USR.
- Parquet records and USR datasets must produce the same `id` for the same sequence.

## Metadata policy

- Output metadata is namespaced: `densegen__<key>`.
- Metadata is validated against `src/dnadesign/densegen/src/core/metadata_schema.py` (typed registry).
- Metadata registry includes field descriptions for documentation.
- Parquet metadata stores list/dict values as native list/struct columns (no JSON encoding).
- Core keys (non-exhaustive):
- `densegen__created_at`, `densegen__schema_version`, `densegen__random_seed`
- `densegen__run_id`, `densegen__run_root`, `densegen__run_config_path`, `densegen__run_config_sha256`
- `densegen__policy_sampling`, `densegen__policy_solver`, `densegen__policy_gc_fill`
- `densegen__solver_backend`, `densegen__solver_strategy`, `densegen__solver_options`
- `densegen__solver_strands`
- `densegen__input_type`, `densegen__input_name`, `densegen__input_path` / `input_dataset` / `input_root`
- `densegen__input_mode`, `densegen__input_pwm_ids`, `densegen__input_pwm_*`
- `densegen__fixed_elements`
  - `densegen__gap_fill_*` (gc targets, actual, attempts, relaxed flag)
  - `densegen__library_*` (size, unique TFs, unique sites)
  - `densegen__used_tf_counts` stored as list of `{tf, count}` records
- `densegen__sampling_pool_strategy`, `densegen__sampling_iterative_*`
- `densegen__required_regulators`
- `densegen__min_required_regulators`
- `densegen__min_count_by_regulator` stored as list of `{tf, min_count}`

## Error handling

- Input parsing, sampling coverage, and plotting options are **strict**.
- Unexpected or malformed data **must not** be silently skipped.
- Errors should be precise and actionable (include path/context where possible).
  - Binding-site tables must not contain null/empty values or duplicate regulator/sequence pairs.
  - Sequence libraries (CSV/Parquet/USR) must not contain null or empty sequences.
  - USR inputs/outputs require explicit `root` paths (no implicit defaults).

## Extension rules (pragmatic changes)

- New behavior must be explicit in config and recorded in metadata.
- Prefer adding small, composable functions over monolithic logic.
- Maintain a single source of truth for schema and defaults (`config/`).

## Audit notes (2026-01-14)

Recent strictness improvements:
- `densegen.schema_version` is required and validated (currently `2.1`).
- Promoter motifs are validated for non-empty A/C/G/T and normalized to uppercase.
- Multi-sink alignment checks Parquet IDs even when `deduplicate: false`.
- Plot styling is strict when `seaborn_style: true` (missing styles raise).
- Metadata schema registry includes field descriptions and enforces typed list/struct values in Parquet.

## Audit notes (2026-01-15)

Spec alignment updates:
- Dense-arrays regulator constraints are now wired end‑to‑end (mapping + required/min counts).
- Regulator labels are carried explicitly (no `tf:tfbs` parsing dependency).
- Solver strand mode is exposed in config and recorded in metadata.
- Run-scoped I/O enforced via `densegen.run` (outputs/logs/plots must be inside run root).

Priority backlog (to keep decoupled + testable):
- **USR parity test**: when USR is installed, assert DenseGen canonical IDs match USR outputs.
