---
doc_id: study-rt-lnrna-reader-spop-condition-structure-matrix-route
surface: study-route-detail
study_id: rt_lnrna_sponging_construct_triage
owner: dnadesign-maintainers
last_verified: 2026-07-07
status: materialized
---

## Reader SPOP Condition-Structure Matrix

Use this path for the RT-lnRNA figure that joins Reader SPOP assay evidence to
retron MSD structure thumbnails. Keep the join in the RT-lnRNA study unit
because the figure answers a study-specific cross-source question.

### Ownership

| Layer | Owner | Contract |
| --- | --- | --- |
| SPOP scoring | Reader | `reader.domains.plate_reader.analysis.spop.score_spop_endpoint` |
| Condition-long bridge | RT-lnRNA study | `reader_spop_composite/condition_matrix.py` |
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

### Missing Data Policy

- Missing condition tiles are rendered as masked gray cells, not zero.
- Positive-control columns are dose-specific because historical retron
  experiments use both 20 nm and 200 nm aTc.
- If a variant lacks an MSD thumbnail, keep the row and mark
  `structure_status`, then route to
  `docs/studies/retron_hairpin_design/routes/README.md` only to materialize
  the missing structure asset.
- Do not trigger new MSD design generation from this route.

### Validation

```bash
uv run pytest -q \
  src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/source/test_reader_spop_plan.py \
  src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/source/test_reader_spop_composite.py \
  src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/tests/source/test_variant_genbank_catalog.py

uv run python -m dnadesign.studies.units.rt_lnrna_sponging_construct_triage.variant_genbank_catalog --json
uv run python -m dnadesign.devtools.architecture.boundaries --repo-root .
```
