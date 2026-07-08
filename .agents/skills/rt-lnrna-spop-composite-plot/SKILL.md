---
name: rt-lnrna-spop-composite-plot
description: Route RT-lnRNA Reader SPOP condition-structure artifacts. Use for SPOP heatmaps, condition matrices, or variant rows with MSD structures. Do not use for LatentDNA UMAPs or MSD design.
metadata:
  version: 0.3.0
  category: workflow-automation
  tags: [rt-lnrna, spop, reader, retron, heatmap, study]
  owner: rt_lnrna_sponging_construct_triage
  routing_priority: specific
---
# RT-lnRNA SPOP Composite Plot
## Purpose
Route requests for the RT-lnRNA Reader SPOP condition-structure matrix artifact.
## Scope
In scope:
- Reader SPOP heatmaps, condition-long tables, category bands, MSD primitive columns, and variant rows joined to MSD structure thumbnails.
- Rebuilding the study-owned artifacts, checking missing cells/thumbnails, and explaining normalization.
Out of scope:
- Generic LatentDNA plots.
- Reader SPOP metric implementation.
- Retron MSD design generation.
- Wet-lab retron protocols.
## Workflow
1. Open the route first. It owns the command, outputs, and missing-data policy:
   `docs/studies/rt_lnrna_sponging_construct_triage/routes/reader-spop-condition-structure-matrix.md`.
2. Open the label contract only when condition semantics or positive-control
   handling matter:
   `docs/studies/rt_lnrna_sponging_construct_triage/contexts/reader-spop-label-contract.md`.
3. Inspect `conditions.py` for treatment labels or heatmap order; inspect
   `row_categories.py` for row-family descriptors, colors, or assignments.
4. Route to retron-hairpin only when `retron_structure_thumbnail_manifest.parquet` reports missing thumbnails or primitive/pairing fields need source inspection.
5. Rebuild with:
   `uv run python -m dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_spop_composite.materialize --reader-root ../reader --json`.
6. Validate with the test matrix in `references/test-matrix.md`.
## Guardrails
- Keep plot code inside `src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reader_spop_composite/`.
- Missing condition cells render as white masked cells, not zero.
- Positive-control columns preserve actual aTc dose.
- Heatmap values are Reader SPOP normalized derepression values, not raw
  cross-experiment fluorescence magnitudes.
- Row-family labels and pastel category fills come from `reader_spop_composite/row_categories.py`, not from renderer-local literals.
- Category fills annotate contiguous row families and do not tint heatmap tiles.
- Heatmap tiles are square, the x-axis title is blank, and the y-axis label
  names the lnRNA variants in the retron Eco1 system.
- Low measured values are visually distinct from white missing cells; the
  current scale moves from pastel cold blue through warm tones to orange.
- Structure thumbnails prefer native ViennaRNA SVGs. The composite applies the
  cap-right coordinate orientation and redraws backbone, base-pair, and
  nucleotide-label geometry as aspect-preserving vector primitives with margins trimmed.
- Poor structure resolution usually means the composite SVG contains embedded
  PNG `<image>` blocks for structure rows; heatmaps/colorbars may be images.
- Primitive columns come from retron-hairpin materialized feature CSVs and
  decomposed MSD-region records, not from image parsing.
- The primitive stem length comes from retron-hairpin `pairing_segments`:
  optional stem-extension bp, payload-stem bp, and foldback-stem bp. It is not
  the 4 bp stem-base length and not a payload-only length.
- Pairing-status fields distinguish canonical Watson-Crick segments from
  intentional wobble or mismatch segments. Narrow foldback annotations and
  explicit complement arms are source notes unless they create unbalanced or
  unresolved pairing segments.
- Active MSD source GenBanks live one variant per file under retron-hairpin `source_inputs/variants/`; do not parse the retired monolithic MSD-region GenBank from this plot route.
- Do not import RT-lnRNA plot semantics into generic LatentDNA modules.
- Do not call retron-hairpin design generation from this route.
- Treat stale available-thumbnail paths as contract errors.
## Required Deliverables
- Route selected and source surfaces opened.
- Output root and manifest path.
- Matrix row count, condition count, variant count, and missing-cell count.
- Row category count and any changed category labels.
- Thumbnail availability or exact missing structure route.
- Validation commands run or scoped reason they were not run.
## Trigger Tests
Should trigger:
- "Make the SPOP heatmap with structure thumbnails."
- "Build the condition structure matrix for Reader SPOP."
- "Plot variant rows with ViennaRNA structure next to SPOP conditions."
- "Rebuild the RT-lnRNA Reader SPOP composite plot."

Should not trigger:
- "Run a generic LatentDNA UMAP."
- "Design new retron MSD sequences."
- "Explain the Reader SPOP equation."

## References
- [external-sources.md](references/external-sources.md)
- [test-matrix.md](references/test-matrix.md)
