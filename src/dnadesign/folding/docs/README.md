## Folding Docs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-13

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
- **Understand contracts:** read the contract flow below before adding a new
  producer or backend.

### Related Design Docs

- [Linear ssDNA composition handoff](../../../../docs/dev/plans/cross-tool/linear-ssdna-composition/2026-05-13-generic-linear-ssdna-composition.md):
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
5. Emit `secondary_structure_prediction_v1.json`.
6. When a `sequence_evidence_map_v1` is available, enrich prediction QA with
   cross-copy predicted pairings and intended-pair recovered/missed counts.
7. Optionally publish `viennarna_secondary_structure_svg_v1.json`, native SVG,
   annotated SVG, and an annotation manifest from the successful prediction.

Missing backends are not treated as success. Advisory requests emit
`warning_optional_missing`; required requests fail the run.

### Architecture Boundaries

- Folding consumes assembled sequence artifacts and typed folding requests.
- Folding does not assemble sequences, resolve Cruncher candidates, or own
  workspaces.
- Folding may publish ViennaRNA-native structure SVGs from successful
  predictions. BaseRender remains the linear/component evidence renderer.
- Construct and study-owned compilers call only the public Folding API or the
  `uv run folding` CLI.

### Internal Organization

- `src/api.py` owns request loading, backend preflight, ViennaRNA Python API or
  `RNAfold` CLI execution, and backend-neutral prediction emission.
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
