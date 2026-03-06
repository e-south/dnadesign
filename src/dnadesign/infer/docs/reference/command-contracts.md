## infer command contracts

### infer run

- Accepts `--config` for YAML-driven runs or `--preset` for single preset-driven runs.
- `--dry-run` validates and prints summary without model execution.
- Unknown or missing selected jobs fail fast with non-zero exit.

### infer extract

- Accepts single-output ad-hoc mode (`--fn`, `--format`) or `--preset`.
- Supports ingest from sequences, records jsonl, pt file, and USR dataset.
- `--write-back` only applies to compatible ingest sources and enforces strict write contracts.

### infer generate

- Requires prompt input and generation parameters.
- Output payload must include `gen_seqs`; malformed payloads fail fast.

### infer adapters

- `list` returns registered model ids.
- `fns` returns namespaced function contracts.
- `cache-clear` clears adapter cache used by runtime loaders.

### infer validate

- `validate config` checks root model/job schema and command contract readiness.
- `validate usr` checks dataset/field accessibility for write-back/resume paths.

### no-silent-fallback contract

- Output cardinality mismatches fail immediately.
- Resume-scan parquet read errors fail immediately.
- Unsupported write-back source contracts fail immediately.
