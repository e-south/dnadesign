# Workspace And Demo Guide

**Owner:** dnadesign-maintainers
**Last verified:** 2026-02-27


This guide defines workspace operations and the curated demo entrypoints.

## Workspace Contract

Each workspace contains:
- `job.yaml`
- `inputs/`
- `outputs/`

Operational behavior:
- `job.yaml` relative paths resolve from the workspace root.
- If `results_root` is omitted, runtime defaults to `outputs/`.
- For `images` output with no explicit `dir`, workspace jobs default to `outputs/plots/`.
- `run_report.json` is optional and emitted only when `run.emit_report: true`.

## Workspace Commands

```bash
# Initialize a new BaseRender workspace scaffold.
uv run baserender workspace init demo_run
# List available BaseRender workspaces.
uv run baserender workspace list
# Validate BaseRender job config and input contracts.
uv run baserender job validate --workspace demo_run
# Execute the BaseRender job for the selected workspace.
uv run baserender job run --workspace demo_run

# if workspaces are outside the default root:
uv run baserender job run --workspace demo_run --workspace-root /path/to/workspaces
```

## Curated Demos

### `demo_densegen_render`
- input: `inputs/input.parquet`
- output: PNG files under `outputs/plots/`
- integration contract: `docs/integrations/densegen.md`

### `demo_cruncher_render`
- input: `inputs/elites_showcase_records.parquet`
- output: PDF files under `outputs/plots/`
- integration contract: `docs/integrations/cruncher.md`

Demo packaging rule:
- keep only runtime-essential primitives in `inputs/`
- keep ad-hoc workspaces out of git

## Run Curated Demos

```bash
# Validate BaseRender job config and input contracts.
uv run baserender job validate --workspace demo_densegen_render --workspace-root src/dnadesign/baserender/workspaces
# Execute the BaseRender job for the selected workspace.
uv run baserender job run --workspace demo_densegen_render --workspace-root src/dnadesign/baserender/workspaces

# Validate BaseRender job config and input contracts.
uv run baserender job validate --workspace demo_cruncher_render --workspace-root src/dnadesign/baserender/workspaces
# Execute the BaseRender job for the selected workspace.
uv run baserender job run --workspace demo_cruncher_render --workspace-root src/dnadesign/baserender/workspaces
```
