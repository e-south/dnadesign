# Workspace And Demo Guide

**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-01


This guide defines workspace operations and the curated demo entrypoints.

## Workspace Contract

Each workspace contains:
- `.baserender-workspace`
- `job.yaml`
- `inputs/`
- `outputs/`
- `README.md`

Operational behavior:
- BaseRender only applies workspace output/path scoping when the workspace marker is present; this prevents accidental inference from arbitrary `job.yaml` directories.
- `job.yaml` relative paths resolve from the workspace root.
- `job.yaml` declares an explicit versioned `bundle.path`, such as `outputs/render-v1`.
- For `images` output with no explicit `dir`, files publish under `<bundle.path>/images/`.
- Every successful bundle includes `manifest.json`; existing bundles are immutable.
- `workspace init` creates an empty generic scaffold on purpose; populate `inputs/input.parquet` or edit `job.yaml` before validate/run.

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

# if the workspace root is outside the default <cwd>/workspaces:
# Initialize the workspace under a non-default parent directory.
uv run baserender workspace init --root /path/to/workspaces demo_run
# List workspaces from that same explicit parent directory.
uv run baserender workspace list --root /path/to/workspaces
# Run the selected workspace while pointing the CLI at the same explicit root.
uv run baserender job run --workspace demo_run --workspace-root /path/to/workspaces
```

## Generic Scaffold Vs Demos Vs Cassette Jobs

- `baserender workspace init` creates a generic standalone scaffold. Its `inputs/` directory starts empty.
- The checked-in package demos live under `src/dnadesign/baserender/workspaces/`.
- Cruncher cassette solve/design jobs are not BaseRender workspaces. They are emitted inside the owning Cruncher workspace and should be run by job-file path.

## Curated Demos

### `demo_densegen_render`
- input: `inputs/input.parquet`
- output: PNG files under `outputs/render-v1/plots/`
- integration contract: `docs/integrations/densegen.md`

### `demo_cruncher_render`
- input: `inputs/elites_showcase_records.parquet`
- output: PDF files under `outputs/render-v1/plots/`
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
