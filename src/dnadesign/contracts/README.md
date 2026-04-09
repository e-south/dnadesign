![contracts banner](assets/contracts-banner.svg)

`contracts` publishes shared cross-tool artifact schemas for `dnadesign`.

Use it when producers and consumers need a neutral, versioned contract surface without importing each other's tool internals. The current package covers cassette and YIU visual artifacts shared between `cruncher` and `baserender`, including the nucleotide-evidence contract used by YIU v4.

See the [repository docs index](../../../docs/README.md) for workflow routes and system runbooks.

## Current exports

- `LinearDuplexViewV1`: shared duplex QA contract for cassette visuals
- `HairpinTopologyViewV1`: shared ssDNA hairpin topology contract
- `CassetteViewsManifestV1`: discovery manifest that groups emitted view files and recommended jobs
- `SequenceEvidenceMapV1`: shared nucleotide-evidence contract for YIU and sibling renderers
- `YiuLinearStateV1`: shared linear/state contract for YIU visual publication
- `YiuHairpinTopologyV1`: shared hairpin topology contract for YIU ligation states
- `YiuTopologyCartoonV1`: shared topology/cartoon contract for YIU circular or branched states

## Tests

- `src/dnadesign/contracts/tests/test_visual_contracts.py`
