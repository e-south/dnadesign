---
doc_id: baserender-reference
title: BaseRender reference
owner: dnadesign-maintainers
status: active
last_verified: 2026-07-30
---

# baserender Reference

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-30


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

`BaseRenderJobV3` / `RenderJobV3` keys and owner modules:

| Job key | Purpose | Primary module(s) |
| --- | --- | --- |
| `version` | Contract version gate (`3`) | `src/config/jobs/base_render_v3.py` |
| `contract` | Optional use-case render-contract descriptor | `src/config/job_contracts.py` |
| `results_root` | Output root resolution | `src/config/jobs/base_render_v3.py`, `src/workspaces/` |
| `input` | Source kind/path + adapter contract | `src/config/adapter_contracts.py`, `src/io/`, `src/adapters/` |
| `selection` | Optional subset/ordering overlay | `src/pipeline/transforms.py` |
| `pipeline` | Transform plugin chain | `src/pipeline/` |
| `render` | Renderer + style preset/overrides | `src/render/`, `src/config/style_v1.py` |
| `outputs` | Explicit artifact declaration | `src/config/jobs/base_render_v3.py`, `src/outputs/` |
| `run` | Strictness and optional report emission | `src/execution/`, `src/reporting/` |

## Job Contract (`BaseRenderJobV3` / `RenderJobV3`)

Required top-level keys:
- `version`
- `input`
- `render`
- `outputs`

Optional top-level keys:
- `contract`
- `results_root`
- `selection`
- `pipeline`
- `run`

Contract behavior:
- unknown keys fail at every level
- `outputs` must be non-empty and explicit
- `contract.kind`, when present, must be compatible with `render.renderer`
- non-workspace default output root is `<job_dir>/results`
- workspace `job.yaml` with sibling `inputs/` and `outputs/` defaults to `<workspace>/outputs`
- `src/config/jobs/base_render_v3.py` is the canonical generic orchestration namespace
- `src/config/jobs/sequence_rows_v3.py` remains a sequence-rows compatibility namespace
- the legacy implementation path `src/config/cruncher_showcase_job.py` remains as a compatibility import shim
- compatibility aliases `SequenceRowsJobV3` and `CruncherShowcaseJob` still load the same schema, but `BaseRenderJobV3` / `RenderJobV3` is the canonical public model name

Render-contract descriptors:
- `base_render_job_v3`: generic adapter -> renderer -> output orchestration; accepts all registered renderer families
- `sequence_rows_render_v3`: linear sequence-row visualization; accepts `sequence_rows`
- `usr_genbank_annotation_render_v1`: USR `seq_annot` GenBank feature-overlay visualization; accepts `sequence_rows`
- `nucleotide_evidence_map_render_v3`: nucleotide-level ownership/evidence map visualization; accepts `nucleotide_evidence_map`
- `hairpin_cartoon_render_v3`: hairpin topology cartoon visualization; accepts `hairpin_cartoon`
- `topology_cartoon_render_v3`: explicit segment-topology cartoon visualization; accepts `topology_cartoon`
- `snapback_map_render_v3`: snapback visual-map rendering; accepts `snapback_map`
- legacy aliases `render_job_v3`, `sequence_rows_v3`, and `cruncher_showcase_v3` resolve to the corresponding descriptor and remain compatibility-only

Adapters:
- `densegen_tfbs`
- `usr_genbank_annotations_v1`
- `generic_features`
- `cruncher_best_window`
- `sequence_windows_v1`
- `duplex_sequence_v1`
- `hairpin_topology_v1`
- `yiu_linear_state_v1`
- `yiu_payload_visual_v1`
- `yiu_hairpin_topology_v1`
- `yiu_topology_cartoon_v1`

### USR GenBank Annotations

`usr_genbank_annotations_v1` is a file-contract adapter for USR rows that already
project GenBank annotations into `seq_annot__features`. USR owns the dataset scan
and projection step; BaseRender consumes the resulting parquet and does not inspect
USR overlay directories.

Required adapter columns:
- `sequence`: row sequence
- `annotations`: `seq_annot__features` list of feature mappings with `start_0` and `end_0`

Optional adapter columns:
- `id`: stable row id used in plot/report output
- `overlay_text`: short row label rendered near the sequence
- `video_subtitle`: subtitle text for video frames
- `source_file`: retained as record metadata
- `product_kind`: retained as record metadata

Semantic mapping:
- `role_hint=sigma70_minus35` or label `-35` renders as the upstream sigma-70 core site.
- `role_hint=sigma70_minus10` or label `-10` renders as the downstream sigma-70 core site.
- `role_hint=TFBS` or labels containing `TFBS` render as regulator binding-site features.
- Full upstream source intervals are treated as row provenance and are not rendered as visual annotation features.
- Promoter calls render as filled `Promoter region` interval annotations.
- Operator labels render as filled `Operator site` interval annotations.
- Other GenBank features render as `Additional annotation` only when `include_untyped_features: true`.

Fail-fast behavior:
- malformed annotation payloads, non-integer coordinates, zero-length intervals, and out-of-bounds intervals fail before rendering;
- `min_per_record` enforces a minimum number of rendered features per adapted row;
- `on_invalid_row: skip` is allowed for exploratory jobs, but production workspaces should use `error` with `run.strict: true`.

Input kinds:
- `parquet`
- `json`
- `jsonl`

Renderer families:
- `sequence_rows`
- `nucleotide_evidence_map`
  - currently implemented as a named sequence-row engine variant for payload/evidence-map contracts; the descriptor is still separate so future behavior can diverge without changing producer contracts
- `hairpin_cartoon`
- `topology_cartoon`
  - topology cartoons require explicit segment geometry; zero-length separator spans are ignored, and visible bands must be positive-length
- `snapback_map`

Shared cross-tool contract models live under `dnadesign.contracts.visual`. Cruncher and other producers publish those contracts; BaseRender parses them and adapts them to `Record`.

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

Video:
- `content_fit: native` preserves the rendered canvas scale and stable crop behavior
- `content_fit: fill_width` trims each rendered frame, keeps a small safe gutter, and scales content to fill the video width
- `content_fit: fill_width_per_frame` trims and scales each frame independently; use for variable-length records where per-record readability matters more than stable cross-frame crop geometry
- BaseRender never non-uniformly stretches frames; fixed-size videos may still letterbox when records have different rendered aspect ratios.

Run reports:
- report emission is opt-in (`run.emit_report: true`)
- workspace jobs default to `outputs/run_report.json`
- non-workspace jobs default to `<results_root>/<job_name>/run_report.json`
- there is no required `reports/` directory contract

Publication:
- `run.conflict_policy` defaults to `error`; an existing image, video, or report destination blocks the batch before rendering
- set `run.conflict_policy: replace` only when the caller explicitly intends to replace the complete published batch
- outputs are rendered to sibling staging paths and published only after every declared output succeeds

Default `results_root`:
- workspace jobs: `<workspace>/outputs`
- non-workspace jobs: `<job_dir>/results` (job-local by default)
- API callers can override scope via `caller_root` (`<caller_root>/results`)

## Public API Boundary

Stable API surface:
- `adapt_record`, `adapt_records`
- `validate_job`, `run_job`, `render`
- `validate_render_job`, `run_render_job`
- `validate_sequence_rows_job`, `run_sequence_rows_job`
- `list_adapters`, `get_adapter_descriptor`
- `list_renderers`, `get_renderer_descriptor`
- `list_render_contracts`, `get_render_contract_descriptor`
- `load_record_from_parquet`, `load_records_from_parquet`
- `render_record_figure`, `render_record_grid_figure`, `render_parquet_record_figure`
- `render_sequence_panel_image`, `sequence_panel_config_for_adapter`
- `BaseRenderJobV3`, `RenderJobV3`, `RenderContractDescriptor`
- `SequencePanelConfig`, `SequencePanelDiagnostics`, `SequencePanelImage`
- `Record`, `Feature`, `Effect`, `Display`, `Span`
- `Style`, `resolve_style`, `resolve_preset_path`, `list_style_presets`
- `SchemaError`, `ContractError`, `LayoutError`

Sequence-panel contract:
- contract id: `dnadesign.baserender.sequence_panel.v1`
- version: `BASERENDER_SEQUENCE_PANEL_CONTRACT_VERSION`
- default profile: `promoter_compact_slide.v1`
- supported adapters: `densegen_tfbs` and `usr_genbank_annotations_v1`
- failure mode: unsupported adapters, unknown profiles, malformed annotation rows, and invalid target dimensions raise `SchemaError`

Sequence-panel layout:
- `vertical_anchor="center"` places the midpoint between the forward and reverse sequence rows at the canvas midpoint. Annotation lanes, titles, and legends do not redefine that anchor.
- `title` accepts caller-owned context as plain text. BaseRender renders the text but does not interpret rank, selection, study, or campaign semantics.
- An adapter-provided `display.overlay_text` remains visible beneath a caller title. `SequencePanelDiagnostics.title` and `record_label` report the two inputs separately.
- A bottom legend follows the lowest occupied sequence or annotation row with the profile's declared content gap. Synthetic space used to balance annotation lanes does not add a second visual gap.
- Panel normalization preserves font sizes and aspect ratio. It scales the complete visible envelope only when needed to fit the declared pixel dimensions.

`render(...)` grid default:
- record list input defaults to a single-row layout (`ncols = len(records)`).
- callers can override with `grid.ncols`.
- invalid/unknown `grid` keys fail fast (`SchemaError`).

Compatibility aliases:
- `validate_cruncher_showcase_job`
- `run_cruncher_showcase_job`
- `SequenceRowsJobV3`
- `CruncherShowcaseJob`

Boundary rule:
- supported imports: `dnadesign.baserender`
- unsupported/private imports: `dnadesign.baserender.src.*`

Tool-specific wiring examples live in:
- `docs/integrations/densegen.md`
- `docs/integrations/cruncher.md`
- `docs/integrations/yiu.md`

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
- `src/public/`: stable programmatic entrypoints and API-level argument checks
- `src/config/`: strict schema loading and style resolution
- `src/io/`: row-source readers
- `src/adapters/`: source-contract to `Record` mapping
- `src/pipeline/`: transforms and selection logic
- `src/render/`: figure composition and effect rendering
- `src/outputs/`: image/video emission
- `src/execution/`: job execution orchestration and strict pre-output gating
- `src/reporting/`: run-report model and write path
- `src/runtime/`: runtime bootstrap and built-in registration
- `src/styles/`: curated style helper ownership
- `src/workspaces/`: workspace discovery and scaffolding

## Extension Points

Add an adapter:
1. Implement adapter class (`apply(row, row_index) -> Record`).
2. Register one `AdapterDescriptor` in `src/config/adapter_contracts.py`.
3. Reuse the descriptor through `src/adapters/registry.py`.
4. Add adapter + end-to-end tests.

Add a feature/effect kind:
1. Register strict validation contract in `src/core/registry.py`.
2. Register renderer drawer in `src/render/effects/registry.py`.
3. Add validation and render tests.

Add a renderer:
1. Implement the renderer in `src/render/`.
2. Register one `RendererDescriptor` in `src/render/renderer.py`.
3. Add render-path tests and, when relevant, update the integration docs that map upstream contract kinds to the renderer family.

Add a render contract descriptor:
1. Add one `RenderContractDescriptor` in `src/config/job_contracts.py`.
2. Keep the descriptor use-case-specific enough to name the visual intent, but generic enough to avoid a one-off source-tool contract.
3. Add a schema/API test that the descriptor accepts its renderer and rejects incompatible renderer families.

## Workspace Contract

Workspace scaffold:
- `.baserender-workspace`
- `job.yaml`
- `inputs/`
- `outputs/`

Operational rules:
- workspace mode is explicit; `job.yaml` directories without the marker use normal job-local defaults
- workspace jobs are self-contained and path-local
- curated demo inputs should include only runtime-essential primitives
- ad-hoc scratch workspaces stay out of git
