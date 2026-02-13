# Baserender Architecture Overview

This document is the implementation-oriented map of `dnadesign.baserender`.

Runtime modules live under `src/dnadesign/baserender/src/` (import root
`dnadesign.baserender.src`). Package-level assets remain at
`src/dnadesign/baserender/` (`docs/`, `styles/`, `tests/`, `workspaces/`).

Public integrations should import from `dnadesign.baserender` only. Internal modules
under `dnadesign.baserender.src.*` are non-contractual implementation detail.

## Package boundaries

- `src/core/`: domain contracts only (`Record`, `Feature`, `Effect`, errors, registries).
  - No file IO.
  - No matplotlib.
  - No pyarrow.
- `src/config/`: strict schema parsing and normalization (`cruncher_showcase_job`, `style_v1`).
  - Unknown keys fail.
  - Ambiguous types fail (for example, string booleans).
  - Adapter shape is centralized in `config/adapter_contracts.py`.
- `src/io/`: data-source readers only.
  - Parquet rows in, Python dict rows out.
  - No domain mapping.
- `src/adapters/`: source-specific row mapping into `Record`.
  - Adapter registry is the single source of truth for:
    - adapter construction (`build_adapter`)
    - required row columns (`required_source_columns`)
  - Adapter registry reads contracts from `config/adapter_contracts.py` to avoid schema drift.
- `src/pipeline/`: record-to-record transforms and selection service.
  - `transforms.py`: plugin loading + transform application.
  - `selection.py`: strict CSV selection + overlay labeling.
- `src/render/`: `Record` -> matplotlib figure.
  - Feature/effect kind validation before draw.
  - Unknown kinds fail.
- `src/outputs/`: artifact writing (images/video).
- `src/reporting/`: run report model and serialization.
- `src/api.py`: orchestrator that wires all layers together.
- `src/cli.py`: command surface only.
- `src/workspace.py`: workspace scaffold/discovery/resolution for run-scoped IO.

## Runtime flow

1. Parse Cruncher showcase job config (`config/`).
2. Resolve source columns from adapter registry (`adapters/registry.py`).
3. Read rows (`io/parquet_source.py`).
4. Convert rows to records (`adapters/*`).
5. Apply transforms and selection (`pipeline/*`).
6. Render records (`render/*`).
7. Write declared outputs only (`outputs/*`).
8. Emit run report (`reporting/*`).

When a job is loaded from `job.yaml` and sibling `inputs/` + `outputs/` directories exist,
omitted `results_root` defaults to `<workspace>/outputs`.

## Extension points

### Add a new adapter

1. Implement adapter class with `apply(row, row_index) -> Record`.
2. Register it in `adapters/registry.py`:
   - factory
   - required and optional row column keys
3. Add config schema support in `config/cruncher_showcase_job.py`:
   - adapter kind
   - allowed `columns` keys
   - allowed `policies` keys + value constraints
4. Add adapter tests and one end-to-end smoke test.

### Add a new feature/effect kind

1. Register validation contract in `core/registry.py`.
2. Register effect drawer in `render/effects/registry.py` if visual.
3. Add tests that unknown kind fails and known kind renders.

## Hard rules used by this architecture

- No implicit outputs.
- No silent unknown keys.
- No silent unknown feature/effect kinds.
- No broad error swallowing in core paths.
- No hidden cross-layer logic (for example, adapter knowledge in `api.py`).
- No global output leakage for workspace-scoped runs.
