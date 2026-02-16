# baserender Reference

Single technical reference for operators and integrators.

## Documentation Policy (YAGNI + Kaizen)

- Keep documentation compact and role-oriented:
  - `README.md` for package overview and quickstart
  - `docs/reference.md` for architecture and core contracts
  - `docs/integrations/*.md` for tool-specific schema contracts
  - `docs/demos/workspaces.md` for workspace/demo operations
- Keep tool-specific content out of `README.md`.
- Prefer executable examples (`docs/examples/*.yaml`) over prose-heavy guides.
- Update existing sections before adding parallel docs with overlapping scope.

## Intent

`baserender` converts sequence-oriented records into visual assets (images and optional video) through strict contracts.

Invariants:
- explicit schemas at job, record, and style boundaries
- fail-fast validation on unknown keys and invalid values
- no silent fallback behavior for contract errors
- tool-agnostic render core; tool-specific semantics stay in adapters/transforms

## Operator Lifecycle

1. Validate job schema and paths.
2. Run job (read rows, adapt, transform, select, render, write outputs).
3. Inspect declared artifacts (`images_dir`, optional `video_path`, optional `report_path`).
4. Iterate by editing `job.yaml` style overrides or adapter/pipeline wiring.

Primary commands:
- `baserender job validate ...`
- `baserender job run ...`
- `baserender job normalize ...`

## Config Schema to Package Architecture

`SequenceRowsJobV3` keys and owner modules:

| Job key | Purpose | Primary module(s) |
| --- | --- | --- |
| `version` | Contract version gate (`3`) | `src/config/cruncher_showcase_job.py` |
| `results_root` | Output root resolution | `src/config/cruncher_showcase_job.py`, `src/workspace.py` |
| `input` | Source kind/path + adapter contract | `src/config/adapter_contracts.py`, `src/io/`, `src/adapters/` |
| `selection` | Optional subset/ordering overlay | `src/pipeline/transforms.py` |
| `pipeline` | Transform plugin chain | `src/pipeline/` |
| `render` | Renderer + style preset/overrides | `src/render/`, `src/config/style_v1.py` |
| `outputs` | Explicit artifact declaration | `src/config/cruncher_showcase_job.py`, `src/outputs/` |
| `run` | Strictness and optional report emission | `src/runner.py`, `src/reporting/` |

## Job Contract (`SequenceRowsJobV3`)

Required top-level keys:
- `version`
- `input`
- `render`
- `outputs`

Optional top-level keys:
- `results_root`
- `selection`
- `pipeline`
- `run`

Contract behavior:
- unknown keys fail at every level
- `outputs` must be non-empty and explicit
- non-workspace default output root is `<job_dir>/results`
- workspace `job.yaml` with sibling `inputs/` and `outputs/` defaults to `<workspace>/outputs`

Adapters:
- `densegen_tfbs`
- `generic_features`
- `cruncher_best_window`
- `sequence_windows_v1`

## Record Contract (`Record`)

Renderer input model:
- `id`
- `alphabet`
- `sequence`
- `features[]`
- `effects[]`
- `display`
- `meta`

Validation behavior:
- unknown feature/effect kinds fail
- unknown render-hint keys fail
- shape/type mismatches fail (no implicit coercion)

## Style Contract (`Style v1`)

Effective merge order:
1. `styles/style_v1/presentation_default.yaml`
2. optional preset
3. `render.style.overrides`

Validation behavior:
- unknown style keys fail at all nested levels
- invalid enums/ranges fail
- no best-effort fallback for malformed style values

Notable sequence-tone keys:
- `style.sequence.bold_consensus_bases` enables motif-informed sequence text tone rendering
- `style.sequence.non_consensus_color` defines the light endpoint
- `style.sequence.tone_quantile_low` / `style.sequence.tone_quantile_high` control quantile min-max normalization

For tool-specific style interpretation, see `docs/integrations/cruncher.md`.

## Output and Report Semantics

Images:
- if `outputs` includes `kind: images` with no `dir`, workspace jobs default to `outputs/plots`
- non-workspace jobs default to `<results_root>/<job_name>/images`
- if `dir` is set, it is resolved relative to output root unless absolute

Run reports:
- report emission is opt-in (`run.emit_report: true`)
- workspace jobs default to `outputs/run_report.json`
- non-workspace jobs default to `<results_root>/<job_name>/run_report.json`
- there is no required `reports/` directory contract

Default `results_root`:
- workspace jobs: `<workspace>/outputs`
- non-workspace jobs: `<job_dir>/results` (job-local by default)
- API callers can override scope via `caller_root` (`<caller_root>/results`)

## Public API Boundary

Stable API surface:
- `validate_job`, `run_job`, `render`
- `validate_sequence_rows_job`, `run_sequence_rows_job`
- `load_record_from_parquet`, `load_records_from_parquet`
- `render_record_figure`, `render_record_grid_figure`, `render_parquet_record_figure`
- `Record`, `Feature`, `Effect`, `Display`, `Span`
- `SchemaError`, `ContractError`, `LayoutError`

`render(...)` grid default:
- record list input defaults to a single-row layout (`ncols = len(records)`).
- callers can override with `grid.ncols`.
- invalid/unknown `grid` keys fail fast (`SchemaError`).

Compatibility aliases:
- `validate_cruncher_showcase_job`
- `run_cruncher_showcase_job`

Boundary rule:
- supported imports: `dnadesign.baserender`
- unsupported/private imports: `dnadesign.baserender.src.*`

Tool-specific wiring examples live in:
- `docs/integrations/densegen.md`
- `docs/integrations/cruncher.md`

## Runtime Flow

1. Parse and validate job.
2. Resolve style and runtime paths.
3. Stream rows from input source.
4. Adapt rows to `Record`.
5. Apply transforms and selection.
6. Render records.
7. Write only declared outputs.
8. Optionally write run report.

## Architecture Map

Core package modules and responsibilities:
- `src/api.py`: stable programmatic entrypoints and API-level argument checks
- `src/config/`: strict schema loading and style resolution
- `src/io/`: row-source readers
- `src/adapters/`: source-contract to `Record` mapping
- `src/pipeline/`: transforms and selection logic
- `src/render/`: figure composition and effect rendering
- `src/outputs/`: image/video emission
- `src/reporting/`: run-report model and write path
- `src/workspace.py`: workspace discovery and scaffolding

## Extension Points

Add an adapter:
1. Implement adapter class (`apply(row, row_index) -> Record`).
2. Register in `src/adapters/registry.py`.
3. Add adapter schema in `src/config/adapter_contracts.py`.
4. Add parser acceptance in job config loader.
5. Add adapter + end-to-end tests.

Add a feature/effect kind:
1. Register strict validation contract in `src/core/registry.py`.
2. Register renderer drawer in `src/render/effects/registry.py`.
3. Add validation and render tests.

## Workspace Contract

Workspace scaffold:
- `job.yaml`
- `inputs/`
- `outputs/`

Operational rules:
- workspace jobs are self-contained and path-local
- curated demo inputs should include only runtime-essential primitives
- ad-hoc scratch workspaces stay out of git
