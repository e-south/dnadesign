---
name: rt-lnrna-spop-composite-plot
description: Route RT-lnRNA Reader SPOP heatmap artifacts. Use for SPOP heatmaps, condition-structure matrices, or variant rows with MSD structures. Do not use for LatentDNA UMAPs or MSD design.
metadata:
  version: 0.2.0
  category: workflow-automation
  tags: [rt-lnrna, spop, reader, retron, heatmap, study]
---
# RT-lnRNA SPOP Composite Plot
## Purpose
Route requests for the RT-lnRNA Reader SPOP condition-structure artifact.
## Scope
In scope:
- Reader SPOP heatmaps.
- Condition-long SPOP matrix tables.
- Variant rows joined to MSD structure thumbnails.
- Rebuilding the study-owned composite plot artifacts.
- Checking missing condition tiles and missing thumbnail status.
Out of scope:
- Generic LatentDNA plots.
- Reader SPOP metric implementation.
- Retron MSD design generation.
- Wet-lab retron protocols.
## Workflow
1. Open the route first:
   `docs/studies/rt_lnrna_sponging_construct_triage/routes/reader-spop-condition-structure-matrix.md`.
2. For source/label semantics, open:
   `docs/studies/rt_lnrna_sponging_construct_triage/contexts/reader-spop-label-contract.md`.
3. For structure assets, route to retron-hairpin only when
   `retron_structure_thumbnail_manifest.parquet` has missing thumbnail rows.
4. To rebuild the artifact, use:
   `uv run python -m dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_spop_composite.materialize --reader-root ../reader --json`.
5. Validate with the test matrix in `references/test-matrix.md`.
## Guardrails
- Keep plot code inside
  `src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reader_spop_composite/`.
- Missing condition cells remain missing and render as masked gray, not zero.
- Positive-control columns preserve actual aTc dose.
- Do not import RT-lnRNA plot semantics into generic LatentDNA modules.
- Do not call retron-hairpin design generation from this route.
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
