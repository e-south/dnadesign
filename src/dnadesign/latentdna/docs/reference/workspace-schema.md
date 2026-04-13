# Workspace Schema

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-12

`latentdna.workspace.v1` is the workspace contract for the current tracer-bullet implementation.

Core sections:

- `schema_version`
- `workspace`
- `defaults`
- `sources`
- `metadata`
- `alignments`
- `views`
- `scalars`
- `landmarks`
- `plots`
- `cohorts`
- `exports`
- `notebooks`
- `recipes`
- `deliverables`
- optional `study_binding`

Implemented schema slices:

- Sources: `usr`, `parquet`, and `matrix_bundle`
- Views: source-backed `vector.kind: column` and `vector.kind: bundle_matrix`, plus derived `vector_difference`, `normalize`, `aggregate_by_key`, `apply_reducer`, and `concatenate`
- Alignments: named `intersection` support over `record_key`, `subject_key`, or explicit key columns
- Scalars: `vector_norm`, safe `column_expression`, `select_columns`, `rename_columns`, and `join_tables`
- Landmarks: predicate-selected sets with `centroid`, `medoid`, or `rows` representation declarations
- Plots: named artifact-driven recipes for `projection_scatter`, `projection_grid`, `heatmap`, `distance_scatter`, `distribution`, and `agreement_summary`
- Cohorts: named `kind: column` metadata groupings over declared sources
- Exports: config-declared reducer/table block bundles for `export matrix`, including explicit alignment-backed block projection onto an `alignment_set` row basis
- Snapshots: ad hoc `snapshot build` materialization from declared sources into workspace-owned key ledgers plus metadata companions
- Notebooks: config-backed `artifact_review` marimo apps that load persisted artifacts only
- Recipes: thin DAGs over the currently implemented primitive command set
- Deliverables: user-facing bundles that reference one recipe plus declared prerequisites and outputs
- Study binding: optional read-only link to one checked-in dnadesign study record, including the study dir and readiness vocabulary used by status surfaces

Current runtime limits:

- `defaults.plot_formats` now controls which image files `plot render` writes; the current renderer supports `svg` and `png`.
- `view derive` now covers the full current declared set, but reducer fitting is still PCA-only.
- `view reduce` currently fits PCA only.
- `scalar derive` currently supports `vector_norm`, `column_expression`, `select_columns`, `rename_columns`, and `join_tables`.
- `sample build` currently supports `all`, `random`, `stratified`, `explicit_ids`, `union`, and `intersection`.
- `export matrix` and `export table` currently support `reduced_view` and `table_columns` blocks, plus optional block-level `alignment` projection onto an explicit aligned row basis.
- Landmark distance scoring is implemented against source-backed views.
- `enrich score` currently supports landmark-neighborhood summaries over `kind: column` cohorts only.
- `neighbors fit` currently supports `euclidean` and `cosine` metrics over explicit full-view, sample, or alignment scopes.
- `cluster fit` currently implements deterministic `kmeans` over view-backed full-view, sample, or alignment scopes.
- `agreement compare` now supports kNN-overlap plus optional cluster agreement and landmark-neighborhood overlap; cluster fitting from projections and richer agreement recipes remain deferred.
- `plot render` now supports `projection_scatter`, `projection_grid`, `heatmap`, `distance_scatter`, `distribution`, and `agreement_summary`.
- `plot render` now accepts either a named `plots.<id>` recipe with no inline plot flags, or an ad hoc inline spec; mixing the two modes is rejected.
- Projection plot recipes may declare a shared `color_column`, explicit `panel_titles`, and optional highlighted label subsets through `label_column` plus `label_values`.
- `snapshot build` now writes `rows.parquet` for the stable row basis plus `metadata.parquet` for copied metadata columns; recipes and deliverables still use live sources unless the workspace explicitly chooses snapshot-backed flows.
- `notebook generate` currently emits interactive marimo artifact-review apps with inline plot viewing plus a runtime scan over `outputs/latentdna/plots`, so persisted plot artifacts remain viewable even when they are not explicitly declared in `notebooks.<id>.artifacts`; richer notebook template families remain deferred.
- `workspace init --from-study-dir` currently hydrates the checked-in promoter-study committee template by binding `anchor60` to the study's merged-anchor dataset, `ctx1k` to the construct-context dataset, and writing a typed `study_binding` block.
- `validate workspace --deep` currently performs schema-only pressure checks against declared sources, views, cohorts, landmarks, and the bound study directory without loading embedding payloads.
- Deliverable loading now rejects declared outputs that the linked recipe does not actually produce, including config-backed outputs such as `views`, `scalars`, `reducers`, `reduced_views`, and `exports`.
- Deliverable status and run inventory now use recorded input and source digests where available, including export and alignment manifests with explicit path-backed provenance.
- Fixture-scale benchmark coverage now lives under `tests/perf/test_benchmark_harness.py`; this is a smoke harness, not a replacement for live promoter-study pressure runs.
