# Contract Exports

**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-21

`contracts` publishes shared cross-tool artifact schemas for `dnadesign`.

## Current exports

Generic sequence and folding contracts:

- `AnnotatedSequencePartV1`: digest-pinned sequence, physical-posture fields,
  source references, and nested zero-based half-open features for atomic
  placement without producer-specific imports
- `LinearSsdnaCompositionV1`: ordered segment, annotation, repeat, and
  provenance contract for linear ssDNA products
- `SecondaryStructurePredictionRequestV1`: backend-neutral folding request
  contract with explicit DNA/RNA backend policy
- `SecondaryStructurePredictionV1`: backend-neutral folding result contract for
  canonical component-unit secondary-structure predictions
- `AssessmentTargetV1`: exact producer-owned molecular state submitted for
  advisory structure assessment, including state and sequence digests,
  physical posture, and intended coordinate pairs
- `AssessmentTargetSequenceV1`: worker-readable exact sequence artifact whose
  identifier, bytes, and digest replay against the assessment target
- `StructureAssessmentRequestV1`: backend and isolation policy for assessing
  one exact target
- `StructureAssessmentRecordV1`: immutable advisory result that binds the
  target, backend prediction, producer version, and request/prediction digests
- `StructureAssessmentPublicationV1`: create-only publication manifest used to
  verify request, prediction, record, target identity, and the exhaustive
  evidence-file inventory
- `RtPartPublicationV1` / `RtPartV1`: provider-neutral publication envelope
  for opaque, digest-closed RT parts, with explicit producer ownership,
  provider references, and declared CDS/protein lengths; it publishes no
  sequence bytes or provider-internal candidate ids

Visual contracts:

- `LinearDuplexViewV1`: shared duplex QA contract for cassette visuals
- `HairpinTopologyViewV1`: shared ssDNA hairpin topology contract
- `CassetteViewsManifestV1`: discovery manifest that groups emitted view files and recommended jobs
- `SequenceEvidenceMapV1`: shared nucleotide-evidence contract for YIU and sibling renderers
- `ViennaRNAStructureSvgV1`: manifest for ViennaRNA-native structure SVG artifacts and dnadesign annotation metadata
- `CompositionReviewSvgV1`: manifest for two-row composition review SVGs plus high-resolution PNG siblings that combine structure and component-span QA views
- `ScarNickVisualV1`: scar-nick construct, nick-event, motif, and fragment-state rendering contract
- `SnapbackVisualV1`: snapback geometry and sequence rendering contract
- `YiuLinearStateV1`: shared linear/state contract for YIU visual publication
- `YiuHairpinTopologyV1`: shared hairpin topology contract for YIU ligation states
- `YiuPayloadVisualV1`: YIU payload sequence and state rendering contract
- `YiuTopologyCartoonV1`: shared topology/cartoon contract for YIU circular or branched states

Domain-qualified handoff contracts:

- `MsdDesignReferenceV1` / `MsdDesignCatalogV1`: Retron MSD-specific
  design-reference handoff contracts for study/Reader integration. These live
  here because Reader is expected to consume frozen references without parsing
  Construct, Folding, BaseRender, or Cruncher internals.

Domain-qualified contracts are allowed here only when a non-owner consumer
needs a frozen record and the alternative would be parsing a tool or study
internal surface. For the MSD contracts, the current consumer is Reader-facing
Retron study integration; the owner boundary remains the Retron study record,
not Construct, Folding, BaseRender, or Cruncher. The v1 promise is additive
compatibility only; breaking changes require a new version or migration. Move a
domain-qualified contract out of shared contracts if it loses its sibling
consumer, becomes study-only, or starts accumulating behavior instead of record
shape.

`RtPartPublicationV1` is a neutral producer-consumer seam, not an RT registry.
Each provider owns the publication file and only the parts it emits; consumers
compose exact references without copying provider-owned sequence payloads.
Consumers that require sequence bytes must resolve the opaque provider reference
through a provider-owned authority or fail closed.

## Tests

- `src/dnadesign/contracts/tests/test_visual_contracts.py`
- `src/dnadesign/contracts/tests/test_sequence_contracts.py`
- `src/dnadesign/contracts/tests/test_annotated_sequence_part_contract.py`
