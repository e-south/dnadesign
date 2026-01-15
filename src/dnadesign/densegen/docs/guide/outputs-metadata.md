# Outputs and metadata

DenseGen writes USR and/or Parquet outputs with a **shared canonical ID scheme**.

## Output targets

- **Parquet**: dataset directory with `part-*.parquet` files.
- **USR**: Dataset.attach with namespace `densegen`.

When multiple targets are configured, DenseGen asserts all targets are in sync before writing.

## Source field

Every record includes a `source` string:

```
source = densegen:{input_name}:{plan_name}
```

This is separate from metadata and is always present.

## Metadata (namespaced)

All metadata keys are prefixed as `densegen__<key>` in outputs.

**Provenance**
- `densegen__schema_version`
- `densegen__created_at`
- `densegen__run_id`, `densegen__run_root`, `densegen__run_config_path`, `densegen__run_config_sha256`
- `densegen__input_type`
- `densegen__input_name`
- `densegen__input_path` or `densegen__input_dataset` / `densegen__input_root`
- `densegen__input_mode`
- `densegen__input_pwm_ids`, `densegen__input_pwm_*` (for PWM inputs)

**Solver + policy**
- `densegen__solver_backend` (null when `strategy: approximate`)
- `densegen__solver_strategy`
- `densegen__solver_options`
- `densegen__solver_strands`
- `densegen__policy_sampling`, `densegen__policy_solver`, `densegen__policy_gc_fill`

**Library + sampling**
- `densegen__library_size`, `densegen__library_unique_tf_count`, `densegen__library_unique_tfbs_count`
- `densegen__sampling_target_length`, `densegen__sampling_achieved_length`
- `densegen__sampling_relaxed_cap`, `densegen__sampling_final_cap`
- `densegen__sampling_pool_strategy`, `densegen__sampling_iterative_*`

**Constraints + postprocess**
- `densegen__fixed_elements`
- `densegen__promoter_constraint` (name if provided)
- `densegen__gap_fill_*` fields (see reference docs)

**Placement stats**
- `densegen__used_tfbs`, `densegen__used_tfbs_detail`, `densegen__used_tf_counts` (list of `{tf, count}`)
- `densegen__covers_all_tfs_in_solution`, `densegen__min_count_per_tf`
- `densegen__required_regulators`, `densegen__covers_required_regulators`
- `densegen__min_required_regulators`
- `densegen__min_count_by_regulator` (list of `{tf, min_count}`)

## Parquet encoding

List/dict metadata values are stored as native list/struct columns in Parquet (no JSON encoding).
USR attaches serialize list/dict metadata values as JSON strings.
Older JSON-encoded Parquet datasets are not supported; regenerate outputs.
DenseGen will fail fast if a Parquet dataset schema does not match the current metadata registry.
DenseGen also creates a local ID index (`_densegen_ids.sqlite`) in the Parquet output directory to keep deduplication and alignment checks fast.

## Metadata registry

DenseGen validates output metadata against a typed registry in
`src/dnadesign/densegen/src/core/metadata_schema.py` to keep fields stable and
explicit as the schema evolves.
