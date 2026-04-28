# Workspace Schema

**Owner:** dnadesign-maintainers
**Last verified:** 2026-04-28

`latentdna.workspace.v1` is the workspace contract for the current tracer-bullet implementation.
Flattened artifact namespaces now live directly under `outputs/`, including
`outputs/plots`, `outputs/notebooks`, `outputs/exports`, and `outputs/runs`.

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

- Sources: `usr`, `parquet`, `matrix_bundle`, `infer_feature_sidecar`, and `infer_feature_scalar_sidecar`
- Views: source-backed `vector.kind: column` and `vector.kind: bundle_matrix`, plus derived `vector_difference`, `normalize`, `aggregate_by_key`, `apply_reducer`, and `concatenate`
- Alignments: named `intersection` support over `record_key`, `subject_key`, or explicit key columns
- Scalars: `vector_norm`, safe `column_expression`, `select_columns`, `rename_columns`, and `join_tables`
- Landmarks: predicate-selected sets with `centroid`, `medoid`, or `rows` representation declarations
- Plots: named artifact-driven recipes for `projection_scatter`, `projection_grid`, `heatmap`, `distance_scatter`, `xy_scatter`, `distribution`, `curve`, `correspondence_heatmap`, and `agreement_summary`
- Cohorts: named `kind: column` metadata groupings over declared sources
- Exports: config-declared reducer/table block bundles for `export matrix`, including explicit alignment-backed block projection onto an `alignment_set` row basis
- Snapshots: ad hoc `snapshot build` materialization from declared sources into workspace-owned key ledgers plus metadata companions
- Notebooks: config-backed `workspace` marimo apps that load the persisted workspace catalog, artifact manifests, and the notebook control-plane payload under `outputs/notebooks/<id>/controls.json`
- Recipes: thin DAGs over the currently implemented primitive command set
- Deliverables: user-facing bundles that reference one recipe plus declared prerequisites and outputs
- Study binding: optional read-only link to one checked-in dnadesign study record through explicit `study_id` and `docs_root` fields, plus readiness vocabulary used by status surfaces
- Output root: `workspace.output_root` must resolve to `<workspace>/outputs`

Current runtime limits:

- `defaults.plot_formats` now controls which image files `plot render` writes; the current renderer supports `svg`, `pdf`, and `png`.
- `defaults.memory_policy` now enforces workspace-wide warn/fail thresholds for heavy reduction, projection, neighbors, clustering, and export steps; callers must pass `--allow-memory-overage` to cross the fail threshold explicitly.
- `view derive` now covers the full current declared set, but reducer fitting is still PCA-only.
- `view reduce` currently fits PCA only.
- `scalar derive` currently supports `vector_norm`, `column_expression`, `select_columns`, `rename_columns`, and `join_tables`.
- `sample build` currently supports `all`, `random`, `stratified`, `explicit_ids`, `union`, and `intersection`, plus optional `reference_set` preservation for view-backed samples.
- `export matrix` and `export table` currently support `reduced_view` and `table_columns` blocks, plus optional block-level `alignment` projection onto an explicit aligned row basis.
- Landmark distance scoring is implemented against source-backed views.
- `enrich score` currently supports landmark-neighborhood summaries over `kind: column` cohorts only.
- `neighbors fit` currently supports `euclidean` and `cosine` metrics over explicit full-view, sample, alignment, or precomputed `reduced_view` scopes; reduced views are already scope-fixed and cannot be re-scoped with `sample` or `alignment`.
- `cluster fit` currently implements deterministic `kmeans` plus Leiden over view-backed or `reduced_view` matrices; the real promoter-study Leiden route is the reduced-view plus explicit-neighbor-set path, not raw aligned 8k-dimensional views.
- `agreement compare` now supports kNN-overlap plus optional cluster agreement and landmark-neighborhood overlap; richer agreement recipes remain deferred.
- `plot render` now supports `projection_scatter`, `projection_grid`, `heatmap`, `distance_scatter`, `xy_scatter`, `distribution`, `curve`, `correspondence_heatmap`, and `agreement_summary`.
- `plot render` now accepts either a named `plots.<id>` recipe with no inline plot flags, or an ad hoc inline spec; mixing the two modes is rejected.
- Projection plot recipes may declare a shared `color_column`, explicit `panel_titles`, and optional highlighted label subsets through `label_column` plus `label_values`.
- `snapshot build` now writes `rows.parquet` for the stable row basis plus `metadata.parquet` for copied metadata columns; recipes and deliverables still use live sources unless the workspace explicitly chooses snapshot-backed flows.
- `notebook generate` currently emits one workspace notebook surface per declared `notebooks.<id>`, with `notebooks.<id>.default_deliverable` selecting the initial catalog focus while all plot, run, and manifest browsing stays read-only.
- `notebook generate` may return `attention` when the notebook is written before the default deliverable plot exists; the explicit degraded state is persisted and `notebook smoke` remains the gate.
- `notebook generate` now refuses to overwrite or regenerate a notebook when the default deliverable has freshness drift; refresh the deliverable or its linked recipe first so the notebook remains a review surface over fresh artifacts.
- `workspace init --from-study-dir` currently hydrates the checked-in promoter-study pre-assay template by binding `anchor_60bp` to the study's merged-anchor dataset, `full_context_1kb` to the construct-context dataset, and writing a typed `study_binding` block.
- `workspace refresh` clears only workspace-local LatentDNA outputs; it never mutates upstream `usr` datasets.
- `validate workspace --deep` currently performs schema-only pressure checks against declared sources, views, cohorts, landmarks, and the bound study directory without loading embedding payloads.
- `infer_feature_sidecar` sources expose canonical Infer outputs from
  `<usr-dataset>/_derived/infer/feature_aliases.parquet` joined to
  `feature_vectors.parquet`, `_views/sequence_views.parquet`, mutable
  `_views/view_semantics.parquet`, and the owning USR dataset rows. These
  sources make `value` the vector column and keep sequence-view fields such as
  `view_id`, `product_kind`, `orientation`, `recommended_pooling`,
  `source_family`, `selection_basis`, and `view_collections` available as row
  metadata.
- `infer_feature_scalar_sidecar` sources expose canonical Infer scalar outputs
  from `<usr-dataset>/_derived/infer/feature_scalar_aliases.parquet` joined to
  `feature_scalars.parquet`, `_views/sequence_views.parquet`, mutable
  `_views/view_semantics.parquet`, and the owning USR dataset rows. These
  sources make `value` the scalar column for diagnostics such as
  `log_likelihood__mean_per_token`. Missing scalar sidecar files expose a
  zero-row schema so planned scalar sources can coexist with partial feature
  datasets without pretending the scalar evidence is present.
- Source-backed views with `role: planned` or `role: retired` are skipped by
  deep vector-column checks and omitted from notebook geometry controls. This
  is a visible degraded contract for upstream feature gaps; planned or retired
  views must not be treated as materialized evidence until their source sidecars
  contain matching rows.
- When a materialized view artifact was built from an older source/vector
  declaration, deep validation reports `materialized_contract_status:
  stale_source_contract` instead of silently accepting the old rows as current.
  Refresh the view artifact before using downstream plots as current evidence.
- Deliverable loading now rejects declared outputs that the linked recipe does not actually produce, including config-backed outputs such as `views`, `scalars`, `reducers`, `reduced_views`, and `exports`.
- Deliverables must now declare explicit semantic fields in config. The runtime no longer hydrates missing `title`, `summary`, `question`, or `section` from legacy `description` and `kind` fields.
- Deliverable status and run inventory now use recorded input and source digests where available, including export and alignment manifests with explicit path-backed provenance.
- Fixture-scale contract coverage now lives under the checked-in contract and CLI tests for the promoter-study pre-assay template; these are smoke checks, not a replacement for live promoter-study pressure runs.
