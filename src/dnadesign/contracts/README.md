![contracts banner](assets/contracts-banner.svg)

`contracts` publishes shared cross-tool artifact schemas for `dnadesign`.

Use it when producers and consumers need a neutral, versioned contract surface without importing each other's tool internals. The current package covers cassette and YIU visual artifacts shared between `cruncher` and `baserender`, including the nucleotide-evidence contract used by YIU v4, plus generic sequence/folding contracts used by Construct composition. Study-family handoff contracts may live here only when they are explicitly domain-qualified and intended for sibling consumers such as Reader.

See the [repository docs index](../../../docs/README.md) for workflow routes and system runbooks.

## Current exports

- `LinearDuplexViewV1`: shared duplex QA contract for cassette visuals
- `HairpinTopologyViewV1`: shared ssDNA hairpin topology contract
- `CassetteViewsManifestV1`: discovery manifest that groups emitted view files and recommended jobs
- `LinearSsdnaCompositionV1`: generic ordered segment, annotation, repeat, and provenance contract for linear ssDNA products
- `MsdDesignReferenceV1` / `MsdDesignCatalogV1`: Retron MSD-specific design-reference handoff contracts for study/Reader integration
- `SecondaryStructurePredictionRequestV1`: backend-neutral folding request contract with explicit DNA/RNA backend policy
- `SecondaryStructurePredictionV1`: backend-neutral folding result contract for canonical component-unit secondary-structure predictions
- `SequenceEvidenceMapV1`: shared nucleotide-evidence contract for YIU and sibling renderers
- `ViennaRNAStructureSvgV1`: manifest for ViennaRNA-native structure SVG artifacts and dnadesign annotation metadata
- `CompositionReviewSvgV1`: manifest for two-row composition review SVGs that combine structure and component-span QA views
- `YiuLinearStateV1`: shared linear/state contract for YIU visual publication
- `YiuHairpinTopologyV1`: shared hairpin topology contract for YIU ligation states
- `YiuTopologyCartoonV1`: shared topology/cartoon contract for YIU circular or branched states

## Tests

- `src/dnadesign/contracts/tests/test_visual_contracts.py`
- `src/dnadesign/contracts/tests/test_sequence_contracts.py`
