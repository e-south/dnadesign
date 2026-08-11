---
doc_id: baserender-reference
title: BaseRender reference
owner: dnadesign-maintainers
status: active
last_verified: 2026-08-10
---

# BaseRender reference

BaseRender converts typed sequence records into deterministic images. It
validates one job, reads its declared inputs, translates each source record
into a neutral Record, applies optional transforms, renders the selected visual
grammar, and publishes one create-only bundle.

    source record
        -> integration adapter
        -> neutral Record
        -> optional transforms and selection
        -> renderer
        -> image or video bundle + manifest

The producing tool owns calculations, rankings, interpretation, and
statistical plots. BaseRender owns only reusable sequence and
declared-topology drawing.

## Discover capabilities

    uv run baserender catalog
    uv run baserender catalog --json

The catalog is the source for installed adapters, transforms, style profiles,
renderers, and render contracts. Its JSON schema is
dnadesign.baserender.catalog.v1. Catalog inspection does not load the plotting
stack or adapter implementations.

Built-in integrations are composed from a static internal registry. There is
no entry-point discovery. This keeps startup deterministic and avoids a public
plugin protocol before an independently packaged integration needs one.

## Run a job

    uv run baserender job validate path/to/job.yaml
    uv run baserender job run path/to/job.yaml

Validation resolves required paths and rejects unknown keys, incompatible
adapters and renderers, unsupported parameters, and unsafe output paths before
rendering. A run stages every declared artifact, writes manifest.json last,
and publishes the complete directory with one atomic, create-only rename.
Existing bundle paths fail.

Use job normalize --out normalized.yaml when an operator needs to inspect
resolved paths. Normalization writes a new file; it does not edit the source
job.

## RenderJobV4

Required top-level keys:

- version
- bundle
- input
- render
- outputs

Optional keys:

- contract
- selection
- pipeline
- run

| Job key | Meaning | Owner |
| --- | --- | --- |
| version | Requires version 4 | src/config/jobs/base_render_v4.py |
| contract | Narrows the permitted renderer family and resource envelope | src/config/job_contracts.py |
| bundle | Names the create-only publication directory | src/config/jobs/base_render_v4.py |
| input | Declares the source, alphabet, and adapter | src/config/, src/io/, src/integrations/ |
| selection | Selects and orders records | src/pipeline/selection.py |
| pipeline | Applies declared transforms | src/pipeline/, src/integrations/ |
| render | Chooses a renderer and style | src/render/, src/config/style_v1.py |
| outputs | Declares bundle-relative images or video | src/outputs/ |
| run | Controls strict skip behavior | src/execution/ |

src/config/render_job_v4.py parses the complete job. It asks integration
descriptors which adapter columns, policies, transform parameters, and path
parameters are valid. The parser has no producer-specific branches.

### Input

Supported file kinds are parquet, json, and jsonl. The adapter descriptor
defines:

- required and optional source columns;
- accepted adapter configuration and policies;
- supported alphabets and renderers;
- input and output limits;
- source paths that must be resolved during validation.

The complete source is checked against any declared resource envelope before
input.limit or selection is applied.

### Pipeline

Built-in transform names and their allowed parameters appear in the catalog.
Unknown built-in names and parameters fail during job validation.

An in-process caller may declare an explicit module:Class transform. This is a
local Python extension point, not an installable plugin protocol. The class
must implement the transform contract and remains the caller's deployment
responsibility.

Selection is neutral pipeline behavior. It is separate from producer transforms
because record matching and ordering do not interpret scientific meaning.

### Render

A renderer draws facts already present in a Record. It may lay out
nucleotides, annotations, declared pairing edges, and declared topology. It
must not calculate a producer score, infer a ranking, or present a statistical
diagnostic as generic rendering.

Render contracts constrain compatible renderers and may apply resource or
sensitivity policies. Use list_render_contracts() or the CLI catalog instead
of maintaining a separate hard-coded list in clients.

### Outputs

Every output path is relative to bundle.path and confined within it.

- Images may publish to a directory or, when the adapter permits it, one file.
- Video requires ffmpeg.
- BaseRender never stretches a frame non-uniformly.
- Private render contracts preserve owner-only directory and file modes.
- A failed render or publication leaves no partial final bundle.

## Neutral record

Record is the renderer input:

- id
- alphabet
- sequence
- features[]
- effects[]
- display
- meta

Feature and effect registries reject unknown kinds. Their payload contracts
reject unknown fields and invalid coordinates. Adapters should retain
provenance in meta without placing producer logic in the neutral model.

## Styles

Style resolution applies, in order:

1. styles/style_v1/presentation_default.yaml
2. an optional preset
3. render.style.overrides

Unknown keys and invalid values fail. Styles control presentation; record,
fragment, junction, or cohort selection belongs in the job's selection or
renderer options.

Built-in integrations can contribute named style profiles. Discover them with
list_style_profiles() and retrieve a defensive copy with
style_profile_overrides(). The public API remains profile-based rather than
exporting producer-named helpers. Sequence-panel helpers require an explicit
style_profile, so a generic API call cannot silently select a domain-specific
presentation.

## Public Python API

Import from dnadesign.baserender. Imports from dnadesign.baserender.src.* are
private.

Job execution:

- validate_job, run_job, render
- validate_render_job, run_render_job

Capability inspection:

- list_adapters, get_adapter_descriptor
- list_transforms, get_transform_descriptor
- list_style_profiles, get_style_profile_descriptor, style_profile_overrides
- list_renderers, get_renderer_descriptor
- list_render_contracts, get_render_contract_descriptor

Record and figure helpers:

- adapt_record, adapt_records
- load_record_from_parquet, load_records_from_parquet
- render_record_figure, render_record_grid_figure
- render_parquet_record_figure
- render_sequence_panel_image, sequence_panel_config_for_adapter

Public models and errors include RenderJobV4, RenderContractDescriptor, Record,
Feature, Effect, Display, Span, Style, SchemaError, ContractError, LayoutError,
and RenderingError.

The package facade is lazy. Importing dnadesign.baserender or reading its
capability catalog does not import Matplotlib or NumPy. Rendering imports the
plotting stack only when needed.

## Package architecture

| Directory | Responsibility |
| --- | --- |
| src/public/ | Stable Python entrypoints and capability queries |
| src/config/ | Job, render-contract, and style validation |
| src/core/ | Neutral records, errors, registries, and input envelopes |
| src/io/ | Source readers and captured-input handling |
| src/integrations/ | Producer translation, producer transforms, and internal descriptors |
| src/pipeline/ | Producer-neutral transform execution and selection |
| src/render/ | Reusable sequence and declared-topology grammars |
| src/outputs/ | Image and video writers |
| src/execution/ | Validated orchestration and atomic publication |
| src/reporting/ | Run report |
| src/workspaces/ | Workspace discovery and scaffolding |

Neutral layers do not import named integration packages. The integration
registry is the composition root and rejects duplicate adapter, transform,
style-profile, render-contract, and sequence-panel names at import time.

## Add a capability

### Add an upstream adapter or transform

1. Put producer translation under src/integrations/<producer>/.
2. Declare its AdapterDescriptor or TransformDescriptor in that provider.
3. Keep producer imports inside the provider package.
4. Add strict contract, negative-path, and end-to-end job tests.
5. Add or update the matching integration guide.

Do not add a producer branch to the job parser. Do not add entry-point loading
for an integration that ships inside dnadesign.

### Add a renderer

Add a core renderer only when the visual grammar is reusable across producers
or represents declared sequence or topology facts that existing renderers
cannot express. Register one RendererDescriptor and test record compatibility,
resource limits, and output publication.

Keep a plot with producer-specific metrics, thresholds, or comparison semantics
in the producing tool. Sharing Matplotlib does not make a plot a BaseRender
responsibility.

### Add a feature or effect

Register its strict validation contract in src/core/registry.py, its drawer in
src/render/effects/registry.py, and tests for unknown fields, invalid
coordinates, and figure output.

### Add an external integration protocol

Do not add one preemptively. Introduce entry-point packs only after an
integration is independently packaged and the internal descriptor contract has
a second real implementation boundary.

## Workspaces

A workspace contains:

    .baserender-workspace
    job.yaml
    inputs/
    outputs/

Relative paths resolve from the workspace root. Direct job-file execution
resolves from the job directory. See the [workspace guide](demos/workspaces.md)
for scaffold and demo commands, and the [integration index](integrations/README.md)
for producer record contracts.
