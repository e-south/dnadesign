## Folding Docs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-21

Folding is the stateless secondary-structure service for dnadesign. Use it when
a sequence producer has already emitted a folding request or bundle and you
need ViennaRNA preflight, prediction, or annotated structure plots.

`RNAfold` is the ViennaRNA command-line program. `RNA` is the ViennaRNA Python
module used by the uv-managed default backend.

### Choose a Path

- **Producer bundle:** use `--bundle <producer-folding-bundle>` when a
  producer has already written a contract-bearing `manifest.json`.
- **Direct request:** use `--request <request.yaml>` when you have an explicit
  `secondary_structure_prediction_request_v1` file.
- **Plot an existing prediction:** use `plot` with a prediction, assembled
  sequence, and optional `sequence_evidence_map_v1`.
- **Publish an assessment:** use the public Python API when a producer needs a
  replayable advisory record tied to an exact molecular-state digest.
- **Understand contracts:** read the contract flow below before adding a new
  producer or backend.

### Related Design Docs

- [Linear ssDNA composition handoff](../../../../docs/architecture/decisions/adr-0002-generic-linear-ssdna-composition.md):
  Construct, Folding, and BaseRender contract boundaries.
- [Construct composition reference](../../construct/docs/reference/linear-ssdna-composition.md):
  current generic assembly contract and bundle layout.

### Command Routes

```bash
uv run folding preflight --request <request.yaml>
uv run folding run --request <request.yaml>
uv run folding plot \
  --prediction <secondary_structure_prediction_v1.json> \
  --assembled-sequence <assembled_sequence.json> \
  --visual-contract <sequence_evidence_map_v1.json> \
  --output-dir <viennarna-plot-dir>
```

For producer-owned bundles, prefer the manifest-backed surface:

```bash
uv run folding preflight --bundle <producer-folding-bundle>
uv run folding run --bundle <producer-folding-bundle>
uv run folding plot --bundle <producer-folding-bundle>
```

### Where Outputs Live

Folding does not create a workspace. Bundle mode is only a path resolver:
Folding reads the producer bundle's `manifest.json`, consumes artifacts under
the bundle's `folding/` and `visual/` directories, and writes ViennaRNA plot
outputs back under `visual/viennarna_secondary_structure/`.

Bundle manifests must declare a Folding-readable producer contract. New
producers should use `producer_folding_bundle_v1`; Construct composition
bundles remain compatible through
`linear_ssdna_composition_bundle_manifest_v1`.

Direct request mode writes to the paths declared by the request or CLI flags.
Keep those paths producer-owned; do not create long-lived Folding output roots
for ad hoc designs.

### Contract Flow

1. Read the request contract.
2. Resolve the assembled sequence artifact referenced by the request.
3. Check backend availability and output writability.
4. Run the backend when available. Prefer the uv-managed ViennaRNA Python API
   (`backend.interface: python_api`, `python_module: RNA`) for local
   reproducibility; use `backend.interface: cli` for a system-provided
   ViennaRNA `RNAfold` executable when a workflow explicitly needs the CLI
   interface.
5. Emit the backend-neutral `secondary_structure_prediction_v1.json` result.
6. When a `sequence_evidence_map_v1` is available, enrich prediction QA with
   cross-copy predicted pairings and intended-pair recovered/missed counts.
7. Optionally publish `viennarna_secondary_structure_svg_v1.json`, native SVG,
   annotated SVG, and an annotation manifest from the successful prediction.

Missing backends are not treated as success. Advisory requests emit
`warning_optional_missing`; required requests fail the run.

### Advisory Assessment Records

`publish_structure_assessment()` accepts a strict
`StructureAssessmentRequestV1`. Its `AssessmentTargetV1` names the source
state type, schema, identifier, state digest, exact DNA sequence and digest,
physical posture, and any intended coordinate pairs. Folding does not infer
those facts or import the producer's domain package.

The assessment runs in an isolated worker process. The policy timeout applies
to both ViennaRNA interfaces. On POSIX systems, cleanup terminates residual
worker-group descendants after every completion or communication failure, and
timeout cleanup cannot wait indefinitely on inherited pipes. The worker
disables the older low-level CLI deadline so the assessment supervisor is the
only timeout authority. Publication is atomic and create-only. The target and
worker-request artifacts are digest-pinned and semantically replayed against
the high-level request. Every referenced backend log must exist. The manifest
binds the request, worker request, prediction, record, target-state digest,
target-sequence digest, and an exhaustive digest inventory of every published
evidence file. Replay occurs while the publication transaction still owns
rollback authority, with final path identity checked before and after replay.
The verified loader rejects missing or extra files, byte drift, cross-record
identity drift, traversal, symlinked paths, and non-regular filesystem entries.

The emitted `StructureAssessmentRecordV1` always has `authority: advisory`.
It cannot make a HOP design valid, identify an experimental construct, or
establish that a sequence is physically ready for assembly. The low-level
`linear_ssdna` value passed to ViennaRNA is a computational projection; the
assessment target preserves the caller-declared strandedness and topology.

The assessment API currently has no CLI route. Add one only when a concrete
operator workflow needs it; the typed Python surface and persisted record are
the present consumer boundary.

Both ViennaRNA interfaces accept only the optional `temperature_c` backend
parameter. Unknown parameters, nonnumeric temperatures, nonfinite values, and
nonpositive values fail request validation.

ViennaRNA is the only implemented request backend. Add no plugin registry until
a second real backend establishes a shared execution contract.

### Architecture Boundaries

- Folding consumes assembled sequence artifacts and typed folding requests.
- Folding assesses an exact caller-owned state without taking authority over
  that state or its scientific acceptance policy.
- Folding does not assemble sequences, resolve Cruncher candidates, or own
  workspaces.
- Folding may publish ViennaRNA-native structure SVGs from successful
  predictions. BaseRender remains the linear/component evidence renderer.
- Construct and study-owned compilers call only the public Folding API or the
  `uv run folding` CLI.

### Internal Organization

- `src/api.py` owns request loading, backend preflight, ViennaRNA Python API or
  `RNAfold` CLI execution, and backend-neutral prediction emission.
- `src/assessment/` owns target projection, worker isolation, create-only
  assessment publication, and verified replay.
- `src/pairing_qa.py` owns intended-vs-predicted and cross-copy pairing
  summaries.
- `src/viennarna_plot.py` coordinates native ViennaRNA SVG publication.
- `src/viennarna_svg.py` owns SVG DOM annotation, layout normalization, and
  canvas fitting.
- `src/viennarna_ontology.py` and `src/viennarna_summary.py` translate
  caller-provided display profiles into labels, hues, and short summaries.
- `src/rnafold.py` owns stdout-compatible ViennaRNA dot-bracket parsing.

Keep these responsibilities separate so BaseRender remains the linear visual
contract renderer and Construct stays a caller of public folding APIs.
