## infer command contracts

### infer run

- Accepts `--config` for YAML-driven runs or `--preset` for single preset-driven runs.
- Config workflows support `ingest.source` values `usr`, `sequences`, `records`, and `pt_file`.
- For `sequences` and `records`, set `ingest.path` in the job config (path is resolved relative to config directory when not absolute).
- For `pt_file`, `ingest.path` is optional; when omitted, infer uses `<config-dir>/<job_id>.pt`.
- For `usr`, `ingest.path` is invalid and fails validation.
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
- `validate config` requires either `--config` or a `config.yaml` in the current working directory.
- `validate config` rejects unknown schema keys and type mismatches with config exit code (`2`).
- `validate usr` checks dataset/field accessibility for write-back/resume paths.

### infer workspace

- `workspace where` resolves effective workspace root from `--root`, `INFER_WORKSPACE_ROOT`, or repo default.
- `workspace where` resolves the template from `--profile` (`local` or `usr-pressure`) unless `--template` is provided.
- `workspace init` creates `config.yaml`, `inputs/`, and `outputs/logs/ops/audit/` for a workspace id.
- `workspace init` defaults to `--profile local` so the scaffold can run with workspace-local files.
- `workspace init --profile usr-pressure` selects the USR pressure-test template.
- `workspace init` rejects path-like ids and existing workspace directories.

### no-silent-fallback contract

- Output cardinality mismatches fail immediately.
- Resume-scan parquet read errors fail immediately.
- `overwrite=false` rejects writes that would replace existing infer overlay values for the same ids/columns.
- Unsupported write-back source contracts fail immediately.
