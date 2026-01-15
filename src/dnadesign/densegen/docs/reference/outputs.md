# DenseGen Output Formats

DenseGen can emit **USR datasets** and/or **Parquet** datasets. Both formats share
the same canonical ID scheme and metadata semantics.
Parquet is the canonical non‑USR output format (columnar, appendable, analytics‑ready).

## Canonical IDs

- Sequence IDs are computed with USR’s canonical algorithm:
  - `normalize_sequence(sequence, bio_type, alphabet)`
  - `compute_id(bio_type, normalized_sequence)`
- DenseGen implements this locally to avoid a hard dependency on USR.
- Parquet and USR must return the same `id` for the same sequence.
- `bio_type` and `alphabet` are defined once in `output.schema` and shared by all sinks.

## Parquet

Parquet output is written as a **dataset directory** (multiple `part-*.parquet` files) for safe
append behavior. Each row contains required fields (`id`, `sequence`, `bio_type`, `alphabet`, `source`)
plus namespaced `densegen__*` metadata columns.

Behavior:
- If `deduplicate: true`, existing IDs in the dataset are loaded and skipped.
- Parquet requires `pyarrow`; if unavailable, DenseGen fails fast.
- The output `path` must be a directory (no single‑file appends).
- List/dict metadata values are stored as native list/struct columns (no JSON encoding).
- If an existing dataset has a mismatched schema (e.g., legacy JSON metadata), DenseGen fails fast and requires a fresh output path.
- DenseGen maintains a local ID index (`_densegen_ids.sqlite`) to speed deduplication and alignment checks.

## USR

USR output uses `Dataset.attach` with a fixed namespace (`densegen`).
USR integration is optional; if you do not use `output.targets: [usr, ...]`, DenseGen should not import USR code.
USR output requires an explicit `output.usr.root`.
Metadata fields match the Parquet namespaced keys, but are stored in USR’s internal format.
List/dict metadata values are serialized to JSON for USR attaches.
USR output skips any IDs that already exist in `records.parquet` (resume-safe).

When multiple outputs are configured, all sinks must be **in sync** before a run.
If one output already exists and the other does not (or IDs differ), DenseGen fails fast.

## Metadata (common)

Keys are namespaced `densegen__<key>`.

## Source column

The `source` column is always present and encodes provenance as:

```
densegen:{input_name}:{plan_name}
```

**Core + policy**

- `densegen__schema_version`, `densegen__created_at`
- `densegen__run_id`, `densegen__run_root`, `densegen__run_config_path`, `densegen__run_config_sha256`
- `densegen__length` (actual), `densegen__sequence_length` (target)
- `densegen__random_seed`
- `densegen__policy_sampling`, `densegen__policy_solver`, `densegen__policy_gc_fill`
- `densegen__plan`, `densegen__solver_backend` (null when `strategy: approximate`), `densegen__solver_strategy`, `densegen__solver_options`
- `densegen__solver_strands`
- `densegen__compression_ratio`
- `densegen__input_type`, `densegen__input_name`, `densegen__input_mode`
- `densegen__input_path` or `densegen__input_dataset` / `densegen__input_root`
- `densegen__input_pwm_ids`, `densegen__input_pwm_*` (PWM inputs)
- `densegen__fixed_elements`
- `densegen__promoter_constraint` (optional), `densegen__visual` (ASCII layout)

**Library + sampling**

- `densegen__tf_list`, `densegen__tfbs_parts`
- `densegen__used_tfbs`, `densegen__used_tfbs_detail`, `densegen__used_tf_counts` (list of `{tf, count}`), `densegen__used_tf_list`
- `densegen__covers_all_tfs_in_solution`, `densegen__min_count_per_tf`
- `densegen__required_regulators`, `densegen__covers_required_regulators`
- `densegen__min_required_regulators`, `densegen__min_count_by_regulator` (list of `{tf, min_count}`)
- `densegen__library_size`, `densegen__library_unique_tf_count`, `densegen__library_unique_tfbs_count`
- `densegen__sampling_target_length`, `densegen__sampling_achieved_length`
- `densegen__sampling_relaxed_cap`, `densegen__sampling_final_cap`
- `densegen__sampling_pool_strategy`, `densegen__sampling_iterative_*`

**Gap fill**

- `densegen__gap_fill_used`, `densegen__gap_fill_bases`, `densegen__gap_fill_end`
- `densegen__gap_fill_gc_min`, `densegen__gap_fill_gc_max`
- `densegen__gap_fill_gc_target_min`, `densegen__gap_fill_gc_target_max`
- `densegen__gap_fill_gc_actual`, `densegen__gap_fill_relaxed`, `densegen__gap_fill_attempts`

Exact fields may expand over time; avoid hard‑coding unless necessary. List/dict values are stored natively in Parquet.

DenseGen validates output metadata against a typed registry in
`src/dnadesign/densegen/src/core/metadata_schema.py`.
