## `permuter` for agents

Supplement to repo-root `AGENTS.md` with Permuter-specific layout and CLI
surface.

### Key paths
- README: `src/dnadesign/permuter/README.md`
- Docs: `src/dnadesign/permuter/docs/`
- Source internals: `src/dnadesign/permuter/src/`
- Public API facade: `src/dnadesign/permuter/__init__.py`
- Packaged resources: `src/dnadesign/permuter/src/resources/`
- Workspace scopes: `src/dnadesign/permuter/workspaces/<scope>/config.yaml`
- Tests: `src/dnadesign/permuter/tests/`

### Path resolution rules
- Workspace config files must be named `config.yaml`.
- Each workspace scope owns its local `outputs/` directory.
- Supported config tokens: `${WORKSPACE_DIR}`, `${WORKSPACES_DIR}`,
  `${PERMUTER_RESOURCE_DIR}`, environment variables, and `~`.
- `${JOB_DIR}` is intentionally invalid.

### Generated vs hand-edited
- Hand-edited: `workspaces/*/config.yaml`, `workspaces/_shared/inputs/*`,
  docs, source, tests, and packaged resources.
- Generated/run artifacts: `workspaces/*/outputs/**`.
- Do not hand-edit generated Parquet; regenerate via CLI.

### Commands
```bash
uv run permuter --help

uv run permuter workspace list --root src/dnadesign/permuter/workspaces
uv run permuter workspace validate --workspace src/dnadesign/permuter/workspaces/nt_scan_demo

uv run permuter run --workspace nt_scan_demo --ref <ref_name>
uv run permuter evaluate --workspace nt_scan_demo --ref <ref_name> --with smoke:placeholder:log_likelihood
uv run permuter plot --workspace nt_scan_demo --ref <ref_name> --metric-id smoke
uv run permuter validate --data src/dnadesign/permuter/workspaces/nt_scan_demo/outputs/records.parquet
```

### Tests

```bash
uv run pytest -q src/dnadesign/permuter/tests
```
