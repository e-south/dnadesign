# baserender

Contract-first sequence rendering with strict schemas and adapter-based integration.

`baserender` exists to turn sequence-oriented records into visual assets with explicit invariants:
- fail fast on invalid config/contract input
- no silent fallback behavior
- small stable public API at `dnadesign.baserender`
- tool-agnostic render core (Cruncher/DenseGen integration via adapters and plugins)

## Table Of Contents

- [Quick Start](#quick-start)
- [Public API](#public-api)
- [CLI Surface](#cli-surface)
- [Contract Inputs](#contract-inputs)
- [Documentation Map](#documentation-map)

## Quick Start

From repo root:

```bash
uv run baserender job validate --workspace demo_densegen_render --workspace-root src/dnadesign/baserender/workspaces
uv run baserender job run --workspace demo_densegen_render --workspace-root src/dnadesign/baserender/workspaces

uv run baserender job validate --workspace demo_cruncher_render --workspace-root src/dnadesign/baserender/workspaces
uv run baserender job run --workspace demo_cruncher_render --workspace-root src/dnadesign/baserender/workspaces
```

`demo_cruncher_render` uses the fast normalized-record hotpath:
- `inputs/elites_showcase_records.parquet`
- `inputs/motif_library.json`

## Public API

Use package-root imports only:

```python
import dnadesign.baserender as br

job = br.validate_job("job.yaml")
report = br.run_job("job.yaml")
fig = br.render(record_or_records)
```

Primary entrypoints:
- `validate_job(path_or_dict, kind=..., caller_root=...)`
- `run_job(path_or_dict, kind=..., strict=..., caller_root=...)`
- `render(record_or_records, renderer=..., style=..., grid=...)`

Compatibility aliases are still exported:
- `validate_cruncher_showcase_job`
- `run_cruncher_showcase_job`

Do not import from `dnadesign.baserender.src.*`.

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

## Contract Inputs

Supported adapter contracts:
- `densegen_tfbs`
- `generic_features`
- `cruncher_best_window`
- `sequence_windows_v1`

Rule: adapters normalize source rows into `Record`; renderer remains contract-driven and tool-agnostic.

## Documentation Map

Keep docs compact and progressive:
- Technical reference and architecture: `docs/reference.md`
- Workspace/demo guide: `docs/demos/workspaces.md`
- Executable job examples: `docs/examples/*.yaml`
