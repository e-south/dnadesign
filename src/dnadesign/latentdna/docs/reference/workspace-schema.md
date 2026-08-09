# Workspace Schema

**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-08

`latentdna.workspace.v1` is the workspace contract for the current tracer-bullet implementation.
Flattened artifact namespaces now live directly under `outputs/`, including
`outputs/plots`, `outputs/notebooks`, `outputs/exports`, and `outputs/runs`.
LatentDNA core contracts are sequence-family agnostic: promoter labels,
RegulonDB annotations, stress-study design axes, or any future biological
sequence-family semantics must enter through workspace configuration, not
package-level defaults.

Core sections:

- `schema_version`
- `workspace`
- `defaults`
- `sources`
- `metadata`
- `metric_definitions`
- `alignments`
- `views`
- `scalars`
- `landmarks`
- `candidate_sets`
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
- Candidate sets: named, study-agnostic groups of view ids or tag-selected views for notebook layouts and representation status surfaces
- Plots: named artifact-driven recipes for `projection_scatter`, `projection_grid`, `heatmap`, `distance_scatter`, `xy_scatter`, `distribution`, `curve`, `correspondence_heatmap`, and `agreement_summary`
- Cohorts: named `kind: column` metadata groupings over declared sources
- Exports: config-declared reducer/table block bundles for `export matrix`,
  `export table`, and `export anndata`, including explicit alignment-backed
  block projection onto an `alignment_set` row basis
- Snapshots: ad hoc `snapshot build` materialization from declared sources into workspace-owned key ledgers plus metadata companions
- Notebooks: config-backed `workspace` marimo apps that load the persisted workspace catalog, artifact manifests, and the notebook control-plane payload under `outputs/notebooks/<id>/controls.json`
- Recipes: thin DAGs over the currently implemented primitive command set
- Deliverables: user-facing bundles that reference one recipe plus declared prerequisites and outputs
- Reference sets: optional display `label`, explicit id lists, plus
  selector-backed membership through `where` selectors over row metadata;
  selectors support `equals`, `in_values`, and non-null membership checks.
- Metadata axes: optional `metadata.axes.<id>` declarations bind one metadata
  column to display, ordering, color, ordinal-subset, noncanonical-bucket, and
  metric-label semantics used by plot, notebook, and scalar review surfaces.
- Metric definitions: optional `metric_definitions.<metric_id>` declarations
  define workspace-owned metric ids with display name, mathematical
  definition, metric family, evidence tier, unit, direction, aggregation level,
  and optional task id. Workspaces use this for study-facing metric vocabulary
  that is not part of the LatentDNA global registry.
- Study binding: optional read-only link to an external study through explicit
  `study_id`, `record_root`, and `deliverable_docs_root` fields. `record_root`
  must contain `record/campaign.yaml`, `record/datasets.yaml`,
  `record/status.md`, and `operations/ops.study.yaml`;
  `deliverable_docs_root` is the root used to resolve declared `study:`
  document references. Paths may be absolute or relative to the invoking
  repository, but they are never inferred from a public dnadesign study tree.
- Output root: `workspace.output_root` must resolve to `<workspace>/outputs`

Current runtime limits:

- `defaults.plot_formats` now controls which image files `plot render` writes; the current renderer supports `svg`, `pdf`, and `png`.
- `defaults.memory_policy` now enforces workspace-wide warn/fail thresholds for heavy reduction, projection, neighbors, clustering, and export steps; callers must pass `--allow-memory-overage` to cross the fail threshold explicitly.
- `view materialize` memory preflight estimates streaming batch memory plus a conservative resident-output allowance for disk-backed memmap pages; large materializations should not assume the destination matrix is free merely because it is file-backed.
- `view derive` now covers the full current declared set, but reducer fitting is still PCA-only.
- `view reduce` currently fits PCA only.
- `scalar derive` currently supports `vector_norm`, `column_expression`, `select_columns`, `rename_columns`, and `join_tables`.
- `sample build` currently supports `all`, `random`, `stratified`, `explicit_ids`, `union`, and `intersection`, plus optional `reference_set` preservation for view-backed samples. Reference-set preservation may use explicit ids or selector-backed row membership.
- `export matrix`, `export table`, and `export anndata` currently support
  `reduced_view` and `table_columns` blocks, plus optional block-level
  `alignment` projection onto an explicit aligned row basis. AnnData exports
  can additionally include explicitly requested projection artifacts in `obsm`
  and neighbor distance artifacts in `obsp` when their row ledgers match the
  export basis exactly.
- Landmark distance scoring is implemented against source-backed views.
- `enrich score` currently supports landmark-neighborhood summaries over `kind: column` cohorts only.
- `neighbors fit` currently supports `euclidean` and `cosine` metrics over explicit full-view, sample, alignment, or precomputed `reduced_view` scopes; reduced views are already scope-fixed and cannot be re-scoped with `sample` or `alignment`.
- `cluster fit` currently implements deterministic `kmeans` plus Leiden over view-backed or `reduced_view` matrices; the real promoter-study Leiden route is the reduced-view plus explicit-neighbor-set path, not raw aligned 8k-dimensional views.
- `agreement compare` now supports kNN-overlap plus optional cluster agreement and landmark-neighborhood overlap; richer agreement recipes remain deferred.
- `plot render` now supports `projection_scatter`, `projection_grid`, `heatmap`, `distance_scatter`, `xy_scatter`, `distribution`, `curve`, `correspondence_heatmap`, and `agreement_summary`.
- `plot render` now accepts either a named `plots.<id>` recipe with no inline plot flags, or an ad hoc inline spec; mixing the two modes is rejected.
- Projection plot recipes may declare a shared `color_column`, explicit
  `panel_titles` for grids, a `panel_title` for single projection scatters,
  typed `hue_options`, optional `scale: panel` for continuous hues whose values
  should be normalized within each panel instead of across a mixed-scope grid,
  and optional highlighted label subsets through `label_column` plus
  `label_values`. Static projection scatters use continuous colorbars for
  `type: continuous` hues and reserve figure space for categorical legends
  instead of drawing legends over the axes.
- Projection annotations may use `plots.<id>.annotation.reference_set` to
  resolve a workspace reference set per panel. Render manifests record
  `expected_ids`, `matched_ids`, and completeness for each panel. Reference sets
  with `label_mode: highlight_only` are drawn as unlabeled overlay markers so
  larger promoter-standard collections can be represented without dense static
  labels.
- `metadata.derivations` supports row-local `copy`, `regex_capture`,
  `map_values`, `coalesce`, and `constant`, row-backed `annotation`, plus
  cross-source `lookup`. `annotation` derivations declare `source: row`, a
  handler path, a study-specific `derive` value, `required_columns`,
  optional `any_required_column_groups`, `missing_policy`, and optional
  `value_type`. Promoter and RegulonDB-specific annotations are explicit
  workspace derivations; generic row contracts do not infer them from column
  names.
  Lookup derivations join a materialized source row's `left_key` to a declared
  source table's `right_key` and copy one `value_column`, failing by default
  on null keys, duplicate right keys, or missing matches.
- Source declarations may add `metadata_include` for source-local row
  metadata. By default those columns append to workspace-level
  `metadata.include`; set `metadata_include_mode: replace` when a source is a
  reference or diagnostic surface with its own explicit metadata boundary and
  should not inherit study-specific annotation derivations from the workspace
  default list.
- `metadata.axes` is the runtime styling contract for categorical, binary, or
  ordinal metadata columns. Each axis declares `column`, optional `label`,
  optional `kind`, `category_order`, `display_labels`,
  `compact_display_labels`, `category_colors`, `ordinal_subset`,
  `metric_labels`, and an optional `noncanonical_policy`. A noncanonical policy
  can provide a hidden or visible bucket plus row selectors that define when a
  category value is canonical. Core plot and notebook code uses this resolved
  axis metadata; it does not infer Sigma-35, promoter, RegulonDB, or other
  study vocabulary from column names.
- `metric_definitions` is the metric-vocabulary contract for study-owned
  scalar ids. The global metric registry contains only package-level generic
  metrics. If a recipe emits a workspace-specific metric id such as a named
  biological axis score, that id must be declared in `metric_definitions`.
  Workspace definitions cannot override global metric ids; use axis
  `metric_labels` or plot labels for presentation changes to generic metrics.
- `snapshot build` now writes `rows.parquet` for the stable row basis plus `metadata.parquet` for copied metadata columns; recipes and deliverables still use live sources unless the workspace explicitly chooses snapshot-backed flows.
- `notebook generate` currently emits one read-only workspace notebook app per
  declared `notebooks.<id>`. The app has one expressive surface selected by an
  artifact-group dropdown rather than top-level review/geometry tabs. Controls
  are data-driven from `controls.json`: persisted plot artifact, candidate-set
  or grid/single mode, model, representation family, context, view, projection,
  hue, and reference overlay. Plot and geometry accordions render in the same
  selected surface so interpretation is not hidden behind tab state.
  `notebook smoke` runs both import smoke and `marimo check` so generated
  notebooks must remain valid marimo DAGs.
- Notebook controls expose both `reference_labels` for small static callout
  sets and `reference_sets` for selector-backed all/collection/subset modes.
  Reference sets may use `where` for OR-style selector membership and
  `where_all` for AND-style membership. Selectors support exact `equals`,
  `in_values`, `regex`, and `not_regex` checks so representation-specific
  suffixes such as core60 or reverse-complement context rows can be included
  without creating separate plot code. Set `notebook_exposed: false` for
  internal preserve sets that should not appear in the annotation dropdown.
  Notebook annotation dropdown labels come from `reference_sets.<id>.label`
  when present; `notebooks.<id>.default_reference_set` selects the initial
  overlay without hard-coding study-specific reference names in runtime code.
- Notebook controls expose `candidate_sets` resolved from workspace
  `candidate_sets` declarations. A candidate set may list explicit `views` or
  select views by exact `include_tags`; optional `panel_titles` are presentation
  labels only. Candidate sets carry each view's role, materialization state, row
  count, dimensionality, and availability so planned output-layer or other
  diagnostic representations can stay visible without being promoted to current
  decision geometry. Fixed-grid notebook presets include only views with at
  least one usable projection artifact by default. Set
  `notebooks.<id>.show_missing_projection_placeholders: true` only when a
  workspace intentionally wants blank missing-projection panels. The
  `candidate_inventory` and candidate-set metadata still list materialized
  output-layer or other non-projected representations for health and alignment
  review.
- Notebook controls also publish the shared `candidate_inventory` ledger used
  by workspace snapshots and catalogs. The generated Marimo runtime reads row
  counts and dimensionality from this control-plane ledger first, then from
  explicit geometry-control rows when the inventory has not been embedded,
  instead of opening view matrices during notebook startup.
- Control-plane builders read materialized view shape from view manifest
  `stats.rows` and `stats.dims`. Current generated workspaces must surface
  missing shape stats as unknown control-plane metadata rather than inspecting
  matrix files during notebook bootstrap.
- Pre-assay `scalar.build` recipes use explicit metadata dimensions for
  `dataset_overview`; package code does not supply promoter-specific default
  dimensions. Recipes that need source class, design family, Sigma-35, or other
  study axes must declare those columns and their category order in params.
- `design_structure_summary` and `context_robustness_summary` are axis-driven:
  workspaces declare `axes` or `retention_axes` entries with `axis_id`,
  `column`, `metric_id`, optional `label`, and optional `exclude_values`.
  The builders do not add design-family, spacer-length, Sigma-35, or other
  cohort metrics unless the recipe declares them.
- Representation scorecards do not add Sigma-35 or other promoter-specific
  neighbor-enrichment metrics by default. Workspaces that need label-enrichment
  summaries declare `neighbor_label_enrichments` params, and study-facing metric
  names live in scalar params plus `metric_definitions`; display-only axis
  labels live in `metadata.axes.<id>.metric_labels`.
- Pre-assay `scalar.build` recipes use the generic `ordinal_axis_audit` builder
  for ordered metadata axes. The axis contract lives in recipe params:
  `axis.column` selects the grouping column, exactly one of `axis.order_path` or
  `axis.rank_column` supplies ranks, `axis.exclude_values` removes controls or
  incompatible labels, and `axis.metric_ids` may map the generic outputs onto a
  study-facing metric vocabulary. Any mapped metric id outside the global
  registry must have a `metric_definitions` entry, keeping study terms such as
  Sigma-35 in workspace config rather than package-level builder selection.
- Centroid-distance heatmaps over ordered or partially ordered categories use
  `axis_centroid_distance` with the same `axis` contract. Ranked values are
  ordered by `axis.order_path` or `axis.rank_column`; unranked observed values
  remain visible after the ranked values instead of being silently dropped.
- `representation_health_summary` computes pairwise cosine-distance summaries
  after the shared standardize/L2-normalize geometry contract. The builder
  evaluates all pairs up to `pairwise_max_rows` rows, defaulting to 4096, and
  otherwise uses a deterministic row sample with `pairwise_seed` so metric
  generation is bounded and reproducible instead of accidentally allocating an
  unbounded dense all-pairs matrix.
- Pre-assay reference-collapse recipes use `reference_alignment_summary` with
  config-declared `reference_sets` when the analysis needs named landmark or
  standard collections. The builder emits group size, median pairwise cosine
  distance, distance IQR, and explicit `reference_set_status` fields such as
  `ok`, `absent`, `missing_rows`, `too_small`, or `missing_columns` instead of
  silently dropping absent collections. `reference_group_columns` remains
  available for broad metadata audits, but named collections should live in
  `reference_sets` so selectors, labels, and notebook overlays share one
  contract.
- Source declarations may carry static sequence semantics:
  `sequence_scope`, `emitted_length_bp`, `source_interval_length_bp`,
  `pooling_span_bp`, `focal_rule`, and `window_selection_rule`.
  Deep validation checks declared fixed emitted lengths and Infer pooling spans
  against observable source metadata. Mixed-length source-insert views are
  allowed, but user-facing scope and panel labels must not claim fixed 60 bp
  windows unless observed lengths are exactly 60 bp.
- `notebook generate` may return `attention` when the notebook is written before the default deliverable plot exists; the explicit degraded state is persisted and `notebook smoke` remains the gate.
- `notebook generate` now refuses to overwrite or regenerate a notebook when the default deliverable has freshness drift; refresh the deliverable or its linked recipe first so the notebook remains a review surface over fresh artifacts.
- `workspace init --from-study-dir` currently hydrates the checked-in promoter-study pre-assay template by binding `merged_anchor_insert` to the study's merged-anchor dataset, `full_context_1kb` to the construct-context dataset, and writing a typed `study_binding` block with separate record-plane and deliverable-doc roots.
- `workspace refresh` clears only workspace-local LatentDNA outputs; it never mutates upstream `usr` datasets.
- `validate workspace --deep` currently performs schema-only pressure checks
  against declared sources, metadata derivation inputs, views, cohorts,
  landmarks, notebooks, and the bound study directory without loading embedding
  payloads. If a source-backed view asks for a derived metadata column, deep
  validation fails unless the derivation and required input columns are
  declared explicitly.
- `infer_feature_sidecar` sources expose canonical Infer outputs from
  `<usr-dataset>/_derived/infer/feature_aliases.parquet` joined to
  `feature_vectors.parquet`, `_views/sequence_views.parquet`, mutable
  `_views/view_semantics.parquet`, and the owning USR dataset rows. These
  sources make `value` the vector column and keep sequence-view fields such as
  `view_id`, `product_kind`, `orientation`, `recommended_pooling`,
  `source_family`, `selection_basis`, and `view_collections` available as row
  metadata. Missing vector sidecar files expose a zero-row schema so planned
  vector sources can be declared before Infer has produced rows. Alias rows
  that omit a vector key or reference absent vector keys still fail validation.
  When multiple alias rows point at the same `feature_vector_key`, LatentDNA
  preserves one row per alias/view and reuses the shared vector payload so row
  metadata such as parent sequence ids cannot be collapsed by feature key.
- `infer_feature_scalar_sidecar` sources expose canonical Infer scalar outputs
  from `<usr-dataset>/_derived/infer/feature_scalar_aliases.parquet` joined to
  `feature_scalars.parquet`, `_views/sequence_views.parquet`, mutable
  `_views/view_semantics.parquet`, and the owning USR dataset rows. These
  sources make `value` the scalar column for diagnostics such as
  `log_likelihood__total` and `log_likelihood__mean_per_token`. Missing scalar
  sidecar files expose a zero-row schema so planned scalar sources can coexist
  with partial feature datasets without pretending the scalar evidence is
  present. As with vector sidecars, aliases that omit a scalar key fail fast,
  and multiple aliases pointing at one `feature_scalar_key` remain separate
  materialized rows with the shared scalar payload.
- Source-backed views with `role: planned` or `role: retired` are skipped by
  deep vector-column checks and omitted from notebook geometry controls. This
  is a visible degraded contract for upstream feature gaps; planned or retired
  views must not be treated as materialized evidence until their source sidecars
  contain matching rows.
- When a materialized view artifact was built from a different source/vector
  declaration, deep validation reports `materialized_contract_status:
  stale_source_contract` instead of silently accepting the old rows as current.
  Refresh the view artifact before using downstream plots as current evidence.
- Deliverable loading now rejects declared outputs that the linked recipe does not actually produce, including config-backed outputs such as `views`, `scalars`, `reducers`, `reduced_views`, and `exports`.
- Deliverables declare `title`, `summary`, `question`, and `section` explicitly;
  `description` and `kind` do not populate missing semantic fields.
- Deliverable status and run inventory now use recorded input and source digests where available, including export and alignment manifests with explicit path-backed provenance.
- Fixture-scale contract coverage now lives under the checked-in contract and CLI tests for the promoter-study pre-assay template; these are smoke checks, not a replacement for live promoter-study pressure runs.
