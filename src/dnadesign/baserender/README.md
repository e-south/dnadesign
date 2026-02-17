# baserender

Contract-first sequence rendering with strict schemas and explicit adapters.

`baserender` turns sequence-oriented records into visual artifacts while enforcing hard contracts:
- fail fast on invalid job/style/record inputs
- no silent fallbacks
- stable public API at `dnadesign.baserender`
- tool-agnostic renderer with adapter-based integration

## What It Is

`baserender` is a composable rendering runtime:
- **render core** handles layout, style, and artifact emission
- **adapters** map source schemas into the canonical `Record` model
- **pipeline transforms** apply optional record shaping before render

Use this package when you want one strict render surface across multiple upstream tools.

## Quick Start

From repo root:

```bash
# Create a new workspace scaffold.
uv run baserender workspace init demo_run

# Validate and run a workspace job.
uv run baserender job validate --workspace demo_run
uv run baserender job run --workspace demo_run
```

## Quick API Usage

Use package-root imports only:

```python
import dnadesign.baserender as br

job = br.validate_job("job.yaml")
report = br.run_job("job.yaml")
figure = br.render(record_or_records)
```

Primary API entrypoints:
- `validate_job(path_or_dict, kind=..., caller_root=...)`
- `run_job(path_or_dict, kind=..., strict=..., caller_root=...)`
- `render(record_or_records, renderer=..., style=..., grid=...)`
- `load_record_from_parquet(...)` / `load_records_from_parquet(...)`
- `render_record_figure(...)` / `render_record_grid_figure(...)` / `render_parquet_record_figure(...)`

`render(...)` grid behavior:
- single record: one panel
- record list: one row by default (`ncols = len(records)`)
- explicit override: `grid={"ncols": <int>}`

Public import boundary:
- supported: `dnadesign.baserender`
- unsupported/private: `dnadesign.baserender.src.*`

## CLI Surface

```bash
baserender job validate <job.yaml>
baserender job run <job.yaml>
baserender job normalize <job.yaml> --out <normalized.yaml>

baserender job validate --workspace <name> [--workspace-root <dir>]
baserender job run --workspace <name> [--workspace-root <dir>]
baserender job normalize --workspace <name> --out <normalized.yaml> [--workspace-root <dir>]

baserender workspace init <name> [--root <dir>]
baserender workspace list [--root <dir>]

baserender style list
baserender style show <preset>
```

## Contracts

Supported adapter contracts:
- `densegen_tfbs`
- `generic_features`
- `cruncher_best_window`
- `sequence_windows_v1`

Rule: adapters normalize source rows into `Record`; rendering remains contract-driven and tool-agnostic.

## Documentation Map

Keep docs compact and role-oriented:
- Technical reference and architecture: `docs/reference.md`
- Tool-specific integration contracts: `docs/integrations/README.md`
- Workspace/demo operations: `docs/demos/workspaces.md`
- Executable job examples: `docs/examples/*.yaml`
