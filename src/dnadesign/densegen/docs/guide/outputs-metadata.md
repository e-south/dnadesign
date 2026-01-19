## Outputs and metadata

DenseGen writes Parquet and/or USR outputs with a shared, deterministic ID scheme. Metadata is
namespaced and recorded consistently so outputs remain resume-safe and auditable.

### Contents
- [Output targets](#output-targets) - Parquet and USR sinks.
- [Run manifest](#run-manifest) - run-level summary JSON.
- [Inputs manifest](#inputs-manifest) - resolved inputs and PWM sampling metadata.
- [Library manifest](#library-manifest) - libraries offered to the solver.
- [Rejection log](#rejection-log) - rejected solutions audit.
- [Source field](#source-field) - per-record provenance string.
- [Metadata scheme](#metadata-scheme) - namespacing and categories.
- [Parquet vs USR encoding](#parquet-vs-usr-encoding) - differences in storage.
- [Metadata registry](#metadata-registry) - canonical schema location.

---

### Output targets

- **Parquet**: dataset directory with `part-*.parquet` files (default, analytics-friendly).
- **USR**: Dataset.attach with namespace `densegen`.

When multiple targets are configured, DenseGen asserts all targets are in sync before writing.

---

### Run manifest

Each run writes `outputs/meta/run_manifest.json` with per-input/plan counts (generated,
duplicates, failures, resamples, libraries built, stalls), plus solver settings, schema version,
and the dense-arrays version source. The manifest also tracks constraint-filter failure reasons
and duplicate-solution counts. A compact `leaderboard_latest` snapshot is recorded per plan
(top TF/TFBS usage, failure hotspots, and diversity coverage) for quick audits without loading
the full outputs.
Use the CLI to summarize a run:

```
uv run dense summarize --run path/to/run
```

---

### Inputs manifest

When a run completes, DenseGen writes `outputs/meta/inputs_manifest.json`. This file captures
the resolved input paths (or dataset roots), PWM sampling settings, and the motif IDs actually
sampled so runs can be audited without re-opening the config or input files. PWM inputs include
per-motif site counts to make sampling behavior explicit.

---

### Run state (checkpoint)

DenseGen writes `outputs/meta/run_state.json` during execution. This checkpoint captures
per-input/plan progress so long runs can resume safely after interruption.

---

### Attempts log

DenseGen writes `outputs/attempts.parquet`, a consolidated log of solver attempts (success,
duplicate, and constraint rejections). Each row includes the attempt status, reason/detail JSON,
the sequence (if available), solver/provenance fields, and the exact library TF/TFBS/site_id lists
offered to the solver. If no attempts occur, the file is absent. Attempts logs use Parquet and
therefore require `pyarrow`.

---

### Site failure summary

DenseGen does not write a separate site-failure table. Per-TFBS failure attribution can be
derived from `outputs/attempts.parquet` by grouping failures over the offered library lists.

---

### Source field

Every record includes a `source` string:

```
source = densegen:{input_name}:{plan_name}
```

This is always present and is separate from metadata.
Detailed placement provenance lives in `densegen__used_tfbs_detail` and the
run-scoped library manifests.

---

### Metadata scheme

All metadata keys are prefixed as `densegen__<key>`.

Typical categories:
- Provenance (`densegen__schema_version`, run identifiers, input info)
- Solver and policy (`densegen__solver_*`, `densegen__policy_*`)
- Library and sampling (`densegen__library_*`, `densegen__sampling_*`, `densegen__sampling_fraction*`)
- Constraints and postprocess (`densegen__fixed_elements`, `densegen__gap_fill_*`)
- Placement stats (`densegen__used_tfbs*`, `densegen__required_regulators*`)

`densegen__sampling_fraction` is defined as the fraction of **unique** TFBS strings in the
solver library divided by the realized input TFBS pool (`input_tfbs_count`).
`densegen__sampling_fraction_pairs` is the fraction of **unique TF:TFBS pairs** in the
solver library divided by `input_tf_tfbs_pair_count` (only meaningful for regulator-bearing inputs).

See `reference/outputs.md` for a fuller list and semantics.

---

### Parquet vs USR encoding

- Parquet stores list/dict metadata as native list/struct columns (no JSON encoding).
- USR stores list/dict metadata as JSON strings in attaches.

DenseGen fails fast if a Parquet dataset schema does not match the current registry.

---

### Metadata registry

DenseGen validates output metadata against a typed registry in
`src/dnadesign/densegen/src/core/metadata_schema.py` to keep fields stable and explicit as the
schema evolves.

---

@e-south
