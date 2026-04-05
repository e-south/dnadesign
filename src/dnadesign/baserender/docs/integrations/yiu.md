# YIU Integration Contract

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-05


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
- `YiuPayloadVisualV1`
- `YiuHairpinTopologyV1`
- `YiuTopologyCartoonV1`

These models are the source of truth for producer and consumer parsing.

## Adapter kinds and renderer mapping

YIU adapter descriptors exposed by `dnadesign.baserender`:

- `yiu_linear_state_v1` -> `sequence_rows`
- `yiu_payload_visual_v1` -> `nucleotide_evidence_map`
- `yiu_hairpin_topology_v1` -> `hairpin_cartoon`
- `yiu_topology_cartoon_v1` -> `topology_cartoon`

Recommended renderer usage:

- `sequence_rows` for linear ssDNA or dsDNA states, retained fragments, and PCR/assembly intermediates
- `nucleotide_evidence_map` for payload-centric YIU contracts that combine mismatch highlighting with PWM motif layers
- `hairpin_cartoon` for ligated ssDNA hairpin states
- `topology_cartoon` for circularized payload candidates and branched/composite topology views
  - topology views must publish explicit segment geometry; baserender does not invent placeholder arms or bands when the contract is incomplete
  - structural zero-length separator spans are ignored by the renderer; visible topology must come from positive-length segments

Payload-centric YIU visuals follow an operator-first strip layout:

- the payload panel emphasizes sequence truth first, then mismatch evidence, then PWM overlays
- split and assembled panels stay diagrammatic and legend-light so the three-view composite reads top-to-bottom
- Cruncher publishes those directions explicitly as `evidence_ribbon` for `payload` and `operator_strip` for `split_payload`/`assembled_payload`; consumers should treat that as producer policy rather than infer it from showcase defaults
- style changes should preserve that information hierarchy instead of adding tool-specific ornament or hidden fallback rendering
- producer-side visual language is owned by Cruncher; `baserender` should consume explicit direction names and style overrides rather than reconstruct the YIU look from consumer-side showcase defaults
- when `yiu_payload_visual_v1` is projected into `sequence_evidence_map_v1`, the projected metadata stays generic (`row_labels`, base highlights, and connector spans only) rather than carrying YIU-namespaced payload metadata into the shared contract

Adapter responsibilities stay split on purpose:

- `yiu_payload_visual_v1.py` orchestrates the public adapter and merges the base sequence projection with motif overlay output.
- `yiu_payload_sequence_projection.py` owns the YIU-to-`sequence_evidence_map_v1` translation and should remain free of motif rendering concerns.
- `yiu_payload_motif_overlay.py` owns motif feature/effect assembly and tag-label enrichment only.
- `yiu_payload_visual_projection.py` is a compatibility facade for callers that need the two helper builders without importing the underlying modules directly.

## Published bundle surface

YIU bundles publish render-facing assets at the bundle root:

- `payload_view.json`
- `split_payload_view.json`
- `assembled_payload_view.json`
- `baserender_jobs/`
- `payload_views.pdf`
- `visual_inventory.json`

Each emitted job is self-contained and resolves paths relative to the owning
bundle. `visual_inventory.json` is the operator-facing inventory.

## Public API examples

Inspect BaseRender support without reaching into private modules:

```python
import dnadesign.baserender as baserender

adapter = baserender.get_adapter_descriptor("yiu_topology_cartoon_v1")
renderer = baserender.get_renderer_descriptor("topology_cartoon")
record = baserender.adapt_record(
    row=payload_contract_mapping,
    adapter_kind="yiu_payload_visual_v1",
    alphabet="IUPAC_DNA",
)
job = baserender.validate_job("baserender_jobs/payload.job.yaml")
report = baserender.run_job("baserender_jobs/payload.job.yaml")
```

CLI path:

```bash
uv run baserender job validate baserender_jobs/payload.job.yaml
uv run baserender job run baserender_jobs/payload.job.yaml
```

Cruncher also exposes a thin public-API wrapper:

```bash
uv run cruncher visuals validate --job baserender_jobs/payload.job.yaml
uv run cruncher visuals run --job baserender_jobs/payload.job.yaml
```

## Boundary rules

- Allowed consumer imports: `dnadesign.baserender`
- Disallowed consumer imports: `dnadesign.baserender.src.*`
- Allowed producer imports for shared schemas: `dnadesign.contracts.visual`
- YIU runtime modules should publish contracts and jobs, not direct render payload calls
