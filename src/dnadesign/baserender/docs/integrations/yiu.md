# YIU Integration Contract

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-27


This page defines the YIU visual-contract handoff used by `baserender`.

## Contract intent

YIU remains responsible for protocol semantics, state derivation, cut geometry,
junction assembly, and admissibility. `baserender` is responsible only for:

- validating shared visual contracts
- adapting those contracts to canonical `Record` instances
- rendering images through explicit renderer families

The default boundary is file-contract-first:

- Cruncher/YIU writes JSON view contracts and sibling `RenderJobV3` YAML files
- `baserender` validates and runs those jobs through public APIs or CLI
- neither side imports the other's private modules

## Shared contract package

YIU contracts are shared through `dnadesign.contracts.visual`:

- `YiuLinearStateV1`
- `YiuHairpinTopologyV1`
- `YiuTopologyCartoonV1`

These models are the source of truth for producer and consumer parsing.

## Adapter kinds and renderer mapping

YIU adapter descriptors exposed by `dnadesign.baserender`:

- `yiu_linear_state_v1` -> `sequence_rows`
- `yiu_hairpin_topology_v1` -> `hairpin_cartoon`
- `yiu_topology_cartoon_v1` -> `topology_cartoon`

Recommended renderer usage:

- `sequence_rows` for linear ssDNA or dsDNA states, retained fragments, and PCR/assembly intermediates
- `hairpin_cartoon` for ligated ssDNA hairpin states
- `topology_cartoon` for circularized payload candidates and branched/composite topology views

## Published bundle surface

YIU bundles publish render-facing assets under:

- `published/views/`
- `published/baserender_jobs/`
- `published/renders/`
- `published/visual_manifest.json`

Each emitted job is self-contained and resolves paths relative to the owning
bundle. `published/visual_manifest.json` is the operator-facing inventory.

## Public API examples

Inspect BaseRender support without reaching into private modules:

```python
import dnadesign.baserender as baserender

adapter = baserender.get_adapter_descriptor("yiu_topology_cartoon_v1")
renderer = baserender.get_renderer_descriptor("topology_cartoon")
job = baserender.validate_job("published/baserender_jobs/circularized_payload_candidate.job.yaml")
report = baserender.run_job("published/baserender_jobs/circularized_payload_candidate.job.yaml")
```

CLI path:

```bash
uv run baserender job validate published/baserender_jobs/circularized_payload_candidate.job.yaml
uv run baserender job run published/baserender_jobs/circularized_payload_candidate.job.yaml
```

Cruncher also exposes a thin public-API wrapper:

```bash
uv run cruncher visuals validate --job published/baserender_jobs/circularized_payload_candidate.job.yaml
uv run cruncher visuals run --job published/baserender_jobs/circularized_payload_candidate.job.yaml
```

## Boundary rules

- Allowed consumer imports: `dnadesign.baserender`
- Disallowed consumer imports: `dnadesign.baserender.src.*`
- Allowed producer imports for shared schemas: `dnadesign.contracts.visual`
- YIU runtime modules should publish contracts and jobs, not direct render payload calls
