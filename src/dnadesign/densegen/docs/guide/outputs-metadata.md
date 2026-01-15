## Outputs and metadata

DenseGen writes Parquet and/or USR outputs with a shared, deterministic ID scheme. Metadata is
namespaced and recorded consistently so outputs remain resume-safe and auditable.

### Contents
- [Output targets](#output-targets) - Parquet and USR sinks.
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

### Source field

Every record includes a `source` string:

```
source = densegen:{input_name}:{plan_name}
```

This is always present and is separate from metadata.

---

### Metadata scheme

All metadata keys are prefixed as `densegen__<key>`.

Typical categories:
- Provenance (`densegen__schema_version`, run identifiers, input info)
- Solver and policy (`densegen__solver_*`, `densegen__policy_*`)
- Library and sampling (`densegen__library_*`, `densegen__sampling_*`)
- Constraints and postprocess (`densegen__fixed_elements`, `densegen__gap_fill_*`)
- Placement stats (`densegen__used_tfbs*`, `densegen__required_regulators*`)

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
