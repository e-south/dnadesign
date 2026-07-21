---
doc_id: study-rt-lnrna-reader-spop-condition-structure-matrix-route
surface: study-route-detail
study_id: rt_lnrna_sponging_construct_triage
owner: dnadesign-maintainers
last_verified: 2026-07-21
status: materialized
---

## Reader SPOP Condition-Structure Matrix

Use this route for the RT-lnRNA figure that joins Reader SPOP assay evidence to
retron MSD structure thumbnails. The join stays in the RT-lnRNA study unit
because the figure answers a study-specific cross-source question.

### Progressive Disclosure

Open surfaces in this order:

1. This route, to confirm ownership, outputs, command, and missing-data policy.
2. `contexts/reader-spop-label-contract.md`, to confirm Reader SPOP label and
   positive-control semantics.
3. `reader_spop_composite/conditions.py`, only when treatment labels, role
   order, or heatmap column order need inspection.
4. `reader_spop_composite/row_categories.py`, only when row-family labels,
   category colors, or variant-to-category assignments need inspection.
5. `reader_spop_composite/structure_manifest.py`, only when thumbnail rows are
   missing, primitive fields are blank, or retron-hairpin source tables fail
   contract checks.
6. `docs/studies/retron_hairpin_design/workbench/provenance/msd_region_records/reader_spop_msd_structure_panel_v1/`,
   only when MSD primitive or pairing fields need inspection.
7. `docs/studies/retron_hairpin_design/...`, only to inspect existing
   materialized structure assets. Do not start MSD design from this route.

### Ownership

| Layer | Owner | Contract |
| --- | --- | --- |
| SPOP scoring | Reader | `reader.domains.plate_reader.analysis.spop.score_spop_endpoint` |
| Condition ontology | RT-lnRNA study | `reader_spop_composite/conditions.py` |
| Row category ontology | RT-lnRNA study | `reader_spop_composite/row_categories.py` |
| Condition-long bridge | RT-lnRNA study | `reader_spop_composite/condition_matrix.py` |
| Parquet table writer | RT-lnRNA study | `reader_spop_composite/tables.py` |
| MSD structure assets | `retron_hairpin_design` study | `reader_spop_msd_structure_panel_v1/materialized` outputs |
| MSD source primitives and pairing facts | `retron_hairpin_design` study | `msd_region_records/reader_spop_msd_structure_panel_v1` |
| MSD primitive visual roles | `retron_hairpin_design` study | `artifact_contracts/primitive_visual_roles.py` |
| Structure thumbnail manifest | RT-lnRNA study | `reader_spop_composite/structure_manifest.py` |
| Structure SVG thumbnail renderer | RT-lnRNA study | `reader_spop_composite/structure_svg.py` |
| Composite plot renderer | RT-lnRNA study | `reader_spop_composite/render.py` |
| Materializer entrypoint | RT-lnRNA study | `reader_spop_composite/materialize.py` |

Do not move this plot into Reader, LatentDNA, or the retron-hairpin study.
Reader owns metric math, LatentDNA owns generic geometry review, and
retron-hairpin owns MSD materialization.

### Data Products

Materialized output root:

```text
docs/studies/rt_lnrna_sponging_construct_triage/workbench/outputs/reader_spop_condition_structure_matrix_v1/
```

Expected files:

```text
tables/reader_spop_condition_matrix.parquet
tables/reader_spop_condition_columns.parquet
tables/retron_structure_thumbnail_manifest.parquet
plots/reader_spop_condition_structure_heatmap.png
plots/reader_spop_condition_structure_heatmap.svg
plots/manifest.json
```

### Plot Contract

- Heatmap tiles are square, with one tile per variant-condition median.
- The plot title is the concise premise
  `Retron edits shift activation and growth`.
- Panel order is `Experiment group`, then `OD600 rel.`, then
  `RFP/OD600 activation`, then `MSD primitives`, then `MSD structure`.
- The `Experiment group` band sits left of the heatmaps and uses rounded
  pastel category blocks for contiguous row families. Category labels include
  `GUU reference`, `tetO HOP`, `Stem-base context`, `Sso7d-RT fusions`,
  `Evo2 RT mutants`, `tetO site swaps`, `Foldback cores`,
  `Stem/cap wobbles`, and `tetO truncations`.
- Row-family colors are annotation only. They do not tint the OD600 or
  activation heatmap tiles.
- MSD structure edge colors come from the retron-hairpin primitive visual-role
  contract through each materialized `annotation_manifest.json`. The RT plot
  does not define cap, payload, foldback, or stem-base colors locally.
- The `OD600 rel.` panel uses the same condition columns as the activation
  heatmap: null, aTc positive control, and IPTG doses when those rows are
  available. It shows Reader's OD600-derived
  `viability_relative_to_baseline` values, so it is a growth proxy relative to
  each observation baseline, not a raw OD600 measurement.
- Both heatmaps show complete condition tick labels and no x-axis title.
- The y-axis label is `lnRNA variants in retron Eco1 system`.
- The plot uses the `publication_dense_v1` typography profile: panel labels,
  variant labels, primitive labels, and colorbar labels are enlarged within the
  square-tile layout rather than by stretching heatmap cells.
- The `MSD primitives` panel displays left base sequence, stem length,
  foldback sequence, and right base sequence from retron-hairpin materialized
  feature CSVs and decomposed MSD-region records. It does not infer these
  values from the rendered image.
- The stem-length column prefers retron-hairpin `pairing_segments`: optional
  stem-extension bp, payload-stem bp, and foldback-stem bp. It is not the
  4 bp stem-base length and not a payload-only length.
- Pairing status fields in `retron_structure_thumbnail_manifest.parquet`
  distinguish canonical Watson-Crick segments from intentional wobble or
  mismatch segments. Noncanonical 170/171 payload pairing is annotation context,
  not a missing-data condition.
- If a pairing segment cannot be balanced or classified, the manifest records a
  `primitive_warning` for review.
- Missing tiles are white. Low or zero measured values are pastel cold blue,
  while higher values move through warm pastel tones to orange. This separates
  missing evidence from low activation.
- Colorbars are skinny, horizontal, and placed in a compact bottom row below
  their corresponding heatmap panels. Each colorbar width matches its heatmap
  width.
- Condition tick labels use compact `aTc` and `IPTG` text for readability; the
  full dose-unit condition keys remain in the manifest and condition-column
  table.
- Values are clipped to the displayed color scale from 0 to 1. Values above
  the aTc positive-control response saturate at the darkest color rather than
  stretching the scale for every rebuild.
- MSD structure thumbnails are rendered larger than the heatmap tiles, without
  a border around the structure column. When native ViennaRNA SVGs are
  available, the renderer applies the cap-right coordinate orientation, keeps
  the native SVG aspect ratio, and redraws backbone, base-pair, and
  nucleotide-label geometry as Matplotlib vector primitives. Structure edge
  colors are resolved from nucleotide spans and basepair roles in the
  retron-hairpin annotation manifest. PNG thumbnails are a fallback only.
- Prominent amber nucleotide markers denote variant text positions that differ
  from the pES-retron-26 structure sequence after pairwise alignment. Deletions
  relative to pES-retron-26 cannot be marked on the variant structure because
  no residue is present at that position.

### Normalization Basis

The heatmap does not show raw RFP/OD600 on an absolute cross-experiment scale.
It shows Reader SPOP normalized derepression values:

- `0 nm aTc; 0 uM IPTG` is the within-observation baseline and maps to 0.
- The observed aTc positive-control condition at `IPTG = 0` maps to 1 while
  preserving the actual aTc dose. The current curated Reader retron benchmark
  set uses `200 nm aTc; 0 uM IPTG`.
- IPTG dose tiles are condition medians. They may be reconstructed from Reader
  normalized endpoint values when the Reader SPOP observation table does not
  carry raw dose-level RFP/OD600 rows.

Use this plot for cross-experiment visual comparison of normalized activation
patterns, not as evidence that absolute fluorescence magnitudes are directly
comparable across experiments.

The adjacent OD600 panel is also normalized within observation: `1.0` means the
condition-level OD600-derived viability estimate matches that variant's own
baseline. Baseline tiles are therefore `1.0`; aTc and IPTG tiles appear when the
Reader artifact carried matching OD600 evidence for those treatments. Read this
panel as compact growth/context evidence beside the SPOP heatmap, not as a
substitute for inspecting raw plate-reader OD600 traces.

### Command

```bash
uv run python -m dnadesign.studies.units.rt_lnrna_sponging_construct_triage.reader_spop_composite.materialize \
  --reader-root ../reader \
  --json
```

The command builds a fresh Reader SPOP plan from the default experiment set,
fails on blocking SPOP plan errors, expands sparse condition rows, resolves
structure thumbnails from the retron-hairpin materialized sequence index, and
writes the plot manifest.

The output contract is file-name stable. Change condition naming, role order,
or Parquet schema through `conditions.py` or `tables.py`; change row-family
labels and assignments through `row_categories.py`; keep plot layout changes
in `render.py`; keep source-asset joins in `structure_manifest.py`.

### Missing Data Policy

- Missing condition tiles are rendered as white masked cells, not zero.
- Positive-control columns are dose-specific. The current curated Reader
  retron benchmark set has one positive-control dose: `200 nm aTc; 0 uM IPTG`.
- If a variant lacks an MSD thumbnail, keep the row and mark
  `structure_status`, then route to
  `docs/studies/retron_hairpin_design/routes/README.md` only to materialize
  the missing structure asset.
- If primitive fields look wrong, inspect the retron-hairpin source record
  first. Active sources are one-variant GenBank files under
  `source_inputs/variants/`, with `variant_sources.yaml` as the source manifest.
  The retired monolithic GenBank is not an active ingest source.
- Current `na` structure rows mean the assay subject is absent from the
  configured retron-hairpin materialized structure source. The default source
  is `reader_spop_msd_structure_panel_v1/materialized`, which provides
  sequence-indexed structures for the 40 Reader SPOP assay subjects.
- If a row claims `structure_status: available` but the referenced PNG or SVG
  path is absent, the renderer fails. Regenerate or repair the thumbnail
  manifest before plotting.
- Do not trigger new MSD design generation from this route.

To make missing structures fetchable or materializable, clarify:

- whether the thumbnail should show MSD-only, full lnRNA, or a named design
  window for each older variant;
- whether older variants should resolve from the RT-lnRNA GenBank/Construct
  authority catalog or from a new retron-hairpin materialization cohort;
- which cohort id and materialized sequence index should be the structure
  source of record;
- how to annotate base-stem-cap orientation for variants that were not produced
  by the current hairpin compiler.

### Validation

```bash
uv run pytest -q \
  src/dnadesign/studies/units/retron_hairpin_design/tests/source_ingest/test_msd_region_genbank.py \
  src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/source/test_reader_spop_plan.py \
  src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/source/test_reader_spop_composite.py \
  src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/source/test_variant_genbank_catalog.py

uv run python -m dnadesign.studies.units.rt_lnrna_sponging_construct_triage.variant_genbank_catalog --json
uv run python -m dnadesign.devtools.architecture.boundaries --repo-root .
```
