## infer workspaces

Use this directory as the default root for infer pressure-test workspaces.

### Quick start

```bash
uv run infer workspace init --id test_stress_ethanol
```

This creates:

- `workspaces/<id>/config.yaml`
- `workspaces/<id>/inputs/`
- `workspaces/<id>/outputs/logs/ops/audit/`

### Contract

- Workspace names must be directory names, not paths.
- Existing workspace directories are never overwritten.
- Config template defaults to:
  - `src/dnadesign/infer/docs/operations/examples/workspace_local_records_config.yaml` (`--profile local`)
- Pressure-test USR template profile:
  - `uv run infer workspace init --id test_stress_ethanol --profile usr-pressure`
  - `src/dnadesign/infer/docs/operations/examples/pressure_test_infer_config.yaml`

### Local data path option (non-USR)

For config-driven local files, set `ingest.source` and `ingest.path` in `config.yaml`:

- `ingest.source: sequences` with `ingest.path: inputs/sequences.txt`
- `ingest.source: records` with `ingest.path: inputs/records.jsonl`
- `ingest.source: pt_file` with optional `ingest.path: inputs/batch.pt`

Relative `ingest.path` values are resolved from the directory that contains `config.yaml`.

### USR reset scope

For USR-backed workspaces, reset only infer outputs with:

```bash
uv run infer prune --usr <dataset-id> --usr-root <usr-root>
```

This archives the `infer` overlay namespace only. It does not delete the workspace or the base USR records table.
