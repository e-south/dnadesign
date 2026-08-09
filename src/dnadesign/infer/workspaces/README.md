## infer workspaces

Infer ships packaged workspace templates and public smoke inputs here. New workspaces created with `infer workspace init` default to `./workspaces/<id>` in the current working directory so the scaffold works both inside and outside a source checkout.

List the packaged workspaces and their current output state with `uv run infer workspace list`.

### Quick start

```bash
uv run infer workspace list # Inspect packaged infer workspaces before choosing one.
uv run infer workspace init --id demo_local_workspace # Create a default infer workspace scaffold under ./workspaces/.
```

This creates:

- `workspaces/<id>/config.yaml`
- `workspaces/<id>/inputs/`
- `workspaces/<id>/outputs/logs/ops/audit/`
- `workspaces/<id>/outputs/usr_datasets/` when `--profile usr-pressure` is used

### Packaged feature-bundle smoke path

- [evo2_feature_bundle_smoke](evo2_feature_bundle_smoke/README.md): small anchor-only plus templated records for the Evo2 feature-bundle contract.

### Contract

- Workspace names must be directory names, not paths.
- Existing workspace directories are never overwritten.
- Config template defaults to:
  - `src/dnadesign/infer/docs/operations/examples/workspace_local_records_config.yaml` (`--profile local`)
- Pressure-test USR template profile:
  - `uv run infer workspace init --id demo_usr_pressure --profile usr-pressure`
  - `src/dnadesign/infer/docs/operations/examples/pressure_test_infer_config.yaml`
  - `ingest.root: outputs/usr_datasets` resolves relative to the workspace `config.yaml`

`outputs/usr_datasets` inside an infer workspace is a workspace-local USR export
root. Use it for self-contained pressure tests or local ownership. Point infer
at an explicit shared USR root when an external workspace owns the cross-tool
dataset.

### Local data path option (non-USR)

For config-driven local files, set `ingest.source` and `ingest.path` in `config.yaml`:

- `ingest.source: sequences` with `ingest.path: inputs/sequences.txt`
- `ingest.source: records` with `ingest.path: inputs/records.jsonl`
- `ingest.source: pt_file` with optional `ingest.path: inputs/batch.pt`

Relative `ingest.path` values are resolved from the directory that contains `config.yaml`.

### USR reset scope

For USR-backed workspaces, reset only infer outputs with:

```bash
uv run infer prune --usr <dataset-id> --usr-root <usr-root> # Archive infer output namespace for one dataset.
```

This archives the `infer` overlay namespace only. It does not delete the workspace or the base USR records table.
