---
doc_id: study-rt-lnrna-reader-spop-condition-structure-matrix-route
surface: study-route-detail
study_id: rt_lnrna_sponging_construct_triage
owner: dnadesign-maintainers
last_verified: 2026-07-07
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
4. `reader_spop_composite/structure_manifest.py`, only when thumbnail rows are
   missing or retron-hairpin source tables fail contract checks.
5. `docs/studies/retron_hairpin_design/...`, only to inspect existing
   materialized structure assets. Do not start MSD design from this route.

### Ownership

| Layer | Owner | Contract |
| --- | --- | --- |
| SPOP scoring | Reader | `reader.domains.plate_reader.analysis.spop.score_spop_endpoint` |
| Condition ontology | RT-lnRNA study | `reader_spop_composite/conditions.py` |
| Condition-long bridge | RT-lnRNA study | `reader_spop_composite/condition_matrix.py` |
| Parquet table writer | RT-lnRNA study | `reader_spop_composite/tables.py` |
| MSD structure assets | `retron_hairpin_design` study | `teto_pwm_trim_rescue_v1` materialized outputs |
| Structure thumbnail manifest | RT-lnRNA study | `reader_spop_composite/structure_manifest.py` |
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
- The x-axis has condition tick labels and no x-axis title.
- The y-axis label is `lnRNA variants in retron Eco1 system`.
- The value palette is white to darker seagreen. White is no activation,
  darker seagreen is higher normalized activation, and missing cells are
  masked gray.
- Values are clipped to the displayed color scale from 0 to 1. Values above
  the aTc positive-control response saturate at the darkest color rather than
  stretching the scale for every rebuild.
- MSD structure thumbnails are rendered larger than the heatmap tiles, without
  a border around the structure column. Near-white thumbnail margins are
  trimmed at render time, then native ViennaRNA thumbnails are rotated 90
  degrees clockwise so the cap faces right in the composite plot.

### Normalization Basis

The heatmap does not show raw RFP/OD600 on an absolute cross-experiment scale.
It shows Reader SPOP normalized derepression values:

- `0 nm aTc; 0 uM IPTG` is the within-observation baseline and maps to 0.
- The observed aTc positive-control condition at `IPTG = 0` maps to 1 while
  preserving the actual aTc dose, so 20 nm and 200 nm aTc remain separate
  columns.
- IPTG dose tiles are condition medians. They may be reconstructed from Reader
  normalized endpoint values when the Reader SPOP observation table does not
  carry raw dose-level RFP/OD600 rows.

Use this plot for cross-experiment visual comparison of normalized activation
patterns, not as evidence that absolute fluorescence magnitudes are directly
comparable across experiments.

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
or Parquet schema through `conditions.py` or `tables.py`; keep plot layout
changes in `render.py`; keep source-asset joins in `structure_manifest.py`.

### Missing Data Policy

- Missing condition tiles are rendered as masked gray cells, not zero.
- Positive-control columns are dose-specific because historical retron
  experiments use both 20 nm and 200 nm aTc.
- If a variant lacks an MSD thumbnail, keep the row and mark
  `structure_status`, then route to
  `docs/studies/retron_hairpin_design/routes/README.md` only to materialize
  the missing structure asset.
- Current `na` structure rows mean the assay subject is absent from the
  configured retron-hairpin materialized `review_variant_ids` index. The
  current `teto_pwm_trim_rescue_v1` hairpin output provides structures for
  retron26, retron43, retron180, and retron195-200.
- If a row claims `structure_status: available` but the PNG path is absent, the
  renderer fails. Regenerate or repair the thumbnail manifest before plotting.
- Do not trigger new MSD design generation from this route.

To make missing structures fetchable or materializable, clarify:

- whether the thumbnail should show MSD-only, full lnRNA, or a named design
  window for each older variant;
- whether older variants should resolve from the RT-lnRNA GenBank/Construct
  authority catalog or from a new retron-hairpin materialization cohort;
- which cohort id and review manifest should be the structure source of record;
- how to annotate base-stem-cap orientation for variants that were not produced
  by the current hairpin compiler.

### Validation

```bash
uv run pytest -q \
  src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/source/test_reader_spop_plan.py \
  src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/source/test_reader_spop_composite.py \
  src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/source/test_variant_genbank_catalog.py

uv run python -m dnadesign.studies.units.rt_lnrna_sponging_construct_triage.variant_genbank_catalog --json
uv run python -m dnadesign.devtools.architecture.boundaries --repo-root .
```
