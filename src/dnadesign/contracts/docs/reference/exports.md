# Contract Exports

**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-26

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
- `SecondaryStructurePredictionV2`: backend-neutral folding result contract for
  canonical component-unit secondary-structure predictions, with mutually
  exclusive structure results and typed execution-failure evidence; V1 result
  artifacts are not accepted as V2
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

Workspace storage contract:

- `dnadesign.contracts.workspace_storage`: strict
  `dnadesign.workspace-storage/v1` parsing and source-closed verification for an
  explicit external workspace root; this envelope owns location, retention,
  and digest identity without interpreting the tool-owned workspace schema

Visual contracts:

- `LinearDuplexViewV1`: shared duplex QA contract for cassette visuals
- `HairpinTopologyViewV1`: shared ssDNA hairpin topology contract
- `CassetteViewsManifestV1`: discovery manifest that groups emitted view files and recommended jobs
- `SequenceEvidenceMapV1`: shared nucleotide-evidence contract for YIU and sibling renderers
- `ViennaRNAStructureSvgV1`: manifest for ViennaRNA-native structure SVG artifacts and dnadesign annotation metadata
- `CompositionReviewSvgV1`: manifest for two-row composition review SVGs plus high-resolution PNG siblings that combine structure and component-span QA views
- `YiuLinearStateV1`: shared linear/state contract for YIU visual publication
- `YiuHairpinTopologyV1`: shared hairpin topology contract for YIU ligation states
- `YiuPayloadVisualV1`: YIU payload sequence and state rendering contract
- `YiuTopologyCartoonV1`: shared topology/cartoon contract for YIU circular or branched states

Domain-qualified contracts belong here only when a demonstrated non-owner
consumer needs a frozen record and the alternative would be parsing a tool or
study internal surface. Study-only records remain with their study owner.

`RtPartPublicationV1` is a neutral producer-consumer seam, not an RT registry.
Each provider owns the publication file and only the parts it emits; consumers
compose exact references without copying provider-owned sequence payloads.
Consumers that require sequence bytes must resolve the opaque provider reference
through a provider-owned authority or fail closed.

## Tests

- `src/dnadesign/contracts/tests/test_visual_contracts.py`
- `src/dnadesign/contracts/tests/test_sequence_contracts.py`
- `src/dnadesign/contracts/tests/test_annotated_sequence_part_contract.py`
