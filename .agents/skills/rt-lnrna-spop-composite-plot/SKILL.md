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
- Reader SPOP heatmaps.
- Condition-long SPOP matrix tables.
- Variant rows joined to MSD structure thumbnails.
- Rebuilding the study-owned composite plot artifacts.
- Checking missing condition tiles and missing thumbnail status.
- Explaining the plot normalization basis and thumbnail materialization status.
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
3. Inspect `reader_spop_composite/conditions.py` only when treatment labels or
   heatmap column order are in scope.
4. Route to retron-hairpin only when
   `retron_structure_thumbnail_manifest.parquet` reports missing thumbnail rows.
   Missing rows mean the variant is absent from the configured hairpin
   materialization manifest, not that the plotter should infer a structure.
5. To rebuild the artifact, use:
   `uv run python -m dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_spop_composite.materialize --reader-root ../reader --json`.
6. Validate with the test matrix in `references/test-matrix.md`.
## Guardrails
- Keep plot code inside
  `src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reader_spop_composite/`.
- Missing condition cells remain missing and render as masked gray, not zero.
- Positive-control columns preserve actual aTc dose.
- Heatmap values are Reader SPOP normalized derepression values, not raw
  cross-experiment fluorescence magnitudes.
- Heatmap tiles are square, the x-axis title is blank, and the y-axis label
  names the lnRNA variants in the retron Eco1 system.
- The color scale is white to darker seagreen with 0 to 1 clipping.
- Structure thumbnails are source images with near-white margins trimmed, then
  rotated 90 degrees clockwise in the composite so the cap points right. Do not
  mutate the hairpin source images.
- Do not import RT-lnRNA plot semantics into generic LatentDNA modules.
- Do not call retron-hairpin design generation from this route.
- Treat stale available-thumbnail paths as contract errors.
## Required Deliverables
- Route selected and source surfaces opened.
- Output root and manifest path.
- Matrix row count, condition count, variant count, and missing-cell count.
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
