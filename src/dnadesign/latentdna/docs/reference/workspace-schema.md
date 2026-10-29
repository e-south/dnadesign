<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[LatentDNA guide](../README.md) / [Reference](README.md)

# Workspace schema

A `config.yaml` with `schema_version: latentdna.workspace.v1` defines the analysis. Generated artifacts live under `<workspace>/outputs`, including plots, notebooks, exports and run records. Inputs and biological labels are declared by the workspace.

Start with the [complete local example](../getting-started/README.md), then add sections as the analysis needs them:

| Purpose | Configuration sections |
| --- | --- |
| Identity and defaults | `schema_version`, `workspace`, `defaults` |
| Inputs and row semantics | `sources`, `metadata`, `alignments`, `views` |
| Comparisons and references | `scalars`, `landmarks`, `cohorts`, `reference_sets`, `candidate_sets`, `metric_definitions` |
| Outputs and execution | `plots`, `exports`, `notebooks`, `recipes`, `deliverables` |
| Optional external study | `study_binding` |

`workspace.output_root` must resolve to `<workspace>/outputs`. A study binding supplies explicit `study_id`, `record_root` and `deliverable_docs_root` paths. The record root must contain `record/campaign.yaml`, `record/datasets.yaml`, `record/status.md` and `operations/ops.study.yaml`; document references beginning with `study:` resolve from `deliverable_docs_root`. See [external study workspaces](../../../../../docs/integrations/study-workspaces.md) for the integration contract.

## Analysis operations

- `defaults.memory_policy` enforces workspace-wide warn/fail thresholds for heavy reduction, projection, neighbors, clustering, and export steps; callers must pass `--allow-memory-overage` to cross the fail threshold explicitly.

- `view materialize` memory preflight estimates streaming batch memory plus a conservative resident-output allowance for disk-backed memmap pages; large materializations should not assume the destination matrix is free merely because it is file-backed.

- `view derive` supports `vector_difference`, `normalize`, `aggregate_by_key`, `apply_reducer` and `concatenate`. See the [view contract](view-contract.md) for coordinate and alignment requirements.

- `view reduce` fits PCA only.

- `scalar derive` supports `vector_norm`, `column_expression`, `select_columns`, `rename_columns`, and `join_tables`.

- `sample build` supports `all`, `random`, `stratified`, `explicit_ids`, `union`, and `intersection`, plus optional `reference_set` preservation for view-backed samples. Reference-set preservation may use explicit ids or selector-backed row membership.

- `export matrix`, `export table`, and `export anndata` support
  `reduced_view` and `table_columns` blocks, plus optional block-level
  `alignment` projection onto an explicit aligned row basis. AnnData exports
  can additionally include explicitly requested projection artifacts in `obsm`
  and neighbor distance artifacts in `obsp` when their row ledgers match the
  export basis exactly.

- Landmark distance scoring is implemented against source-backed views.

- `enrich score` supports landmark-neighborhood summaries over `kind: column` cohorts only.

- `neighbors fit` supports `euclidean` and `cosine` metrics over explicit full-view, sample, alignment, or precomputed `reduced_view` scopes; reduced views are already scope-fixed and cannot be re-scoped with `sample` or `alignment`.

- `cluster fit` implements deterministic `kmeans` plus Leiden over view-backed or `reduced_view` matrices. Use an explicit neighbor set when running Leiden over a reduced view.

- `agreement compare` supports kNN-overlap plus optional cluster agreement and landmark-neighborhood overlap.

- `snapshot build` writes `rows.parquet` for the stable row basis plus `metadata.parquet` for copied metadata columns; recipes and deliverables still use live sources unless the workspace explicitly chooses snapshot-backed flows.

## Sources and sequence semantics

Use the [source reference](source-contract.md) to select the input format and row keys.

- Source declarations may add `metadata_include` for source-local row
  metadata. By default those columns append to workspace-level
  `metadata.include`; set `metadata_include_mode: replace` when a source is a
  reference or diagnostic surface with its own explicit metadata boundary and
  should not inherit study-specific annotation derivations from the workspace
  default list.

- Source declarations may carry static sequence semantics:
  `sequence_scope`, `emitted_length_bp`, `source_interval_length_bp`,
  `pooling_span_bp`, `focal_rule`, and `window_selection_rule`.
  Deep validation checks declared fixed emitted lengths and Infer pooling spans
  against observable source metadata. Mixed-length source-insert views are
  allowed, but user-facing scope and panel labels must not claim a fixed length unless observed lengths match it.

- `validate workspace --deep` performs source-schema checks
  against declared sources, metadata derivation inputs, views, cohorts,
  landmarks, notebooks, and the bound study directory without loading embedding
  payloads. If a source-backed view asks for a derived metadata column, deep
  validation fails unless the derivation and required input columns are
  declared explicitly.

- `infer_feature_sidecar` and `infer_feature_scalar_sidecar` read Infer aliases and committed payloads with SeqLedger row and sequence-view metadata. The vector or scalar column is `value`. See [input sources](source-contract.md#infer-sidecars) for read modes, integrity checks and missing-payload behavior.

- Source-backed views with `role: planned` or `role: retired` are skipped by
  deep vector-column checks and omitted from notebook geometry controls. This
  is an explicit missing-data state for upstream feature gaps; planned or retired
  views must not be treated as materialized evidence until their source sidecars
  contain matching rows.

## Metadata and analysis definitions

- `metadata.derivations` supports row-local `copy`, `regex_capture`,
  `map_values`, `coalesce`, and `constant`, row-backed `annotation`, plus
  cross-source `lookup`. `annotation` derivations declare `source: row`, a
  handler path, a study-specific `derive` value, `required_columns`,
  optional `any_required_column_groups`, `missing_policy`, and optional
  `value_type`. Domain-specific annotations are explicit workspace derivations; they are not inferred from column names.
  External handlers accept a row dictionary and keyword `derive`, and return a
  value or raise `ValueError` for invalid annotations. `missing_policy: null`
  converts that validation failure to a null value; other execution errors
  propagate. Handlers need no LatentDNA imports.
  Lookup derivations join a materialized source row's `left_key` to a declared
  source table's `right_key` and copy one `value_column`, failing by default
  on null keys, duplicate right keys, or missing matches.

- `metadata.axes` is the runtime styling contract for categorical, binary, or
  ordinal metadata columns. Each axis declares `column`, optional `label`,
  optional `kind`, `category_order`, `display_labels`,
  `compact_display_labels`, `category_colors`, `ordinal_subset`,
  `metric_labels`, and an optional `noncanonical_policy`. A noncanonical policy
  can provide a hidden or visible bucket plus row selectors that define when a
  category value is canonical. Core plot and notebook code uses this resolved
  axis metadata; it does not infer biological meaning from column names.

- `metric_definitions` is the metric-vocabulary contract for study-owned
  scalar ids. The global metric registry contains only package-level generic
  metrics. If a recipe emits a workspace-specific metric id such as a named
  biological axis score, that id must be declared in `metric_definitions`.
  Workspace definitions cannot override global metric ids; use axis
  `metric_labels` or plot labels for presentation changes to generic metrics.

- `scalar.build` recipes declare the metadata dimensions for `dataset_overview` explicitly, including the columns and category order.

- `design_structure_summary` and `context_robustness_summary` use declared `axes` or `retention_axes` entries with `axis_id`, `column`, `metric_id`, optional `label` and optional `exclude_values`. Only declared cohort metrics are computed.

- Representation scorecards use explicit `neighbor_label_enrichments` when label enrichment is needed. Metric names belong in scalar parameters and `metric_definitions`; display labels belong in `metadata.axes.<id>.metric_labels`.

- `scalar.build` recipes use the generic `ordinal_axis_audit` builder
  for ordered metadata axes. The axis contract lives in recipe params:
  `axis.column` selects the grouping column, exactly one of `axis.order_path` or
  `axis.rank_column` supplies ranks, `axis.exclude_values` removes controls or
  incompatible labels, and `axis.metric_ids` may map the generic outputs onto a
  study-facing metric vocabulary. Any mapped metric id outside the global
  registry must have a `metric_definitions` entry, keeping domain-specific metric names in workspace configuration.

- Centroid-distance heatmaps over ordered or partially ordered categories use
  `axis_centroid_distance` with the same `axis` contract. Ranked values are
  ordered by `axis.order_path` or `axis.rank_column`; unranked observed values
  remain visible after the ranked values instead of being silently dropped.

- `representation_health_summary` computes pairwise cosine-distance summaries
  after the shared standardize/L2-normalize geometry contract. The builder
  evaluates all pairs up to `pairwise_max_rows` rows, defaulting to 4096, and
  otherwise uses a deterministic row sample with `pairwise_seed` so metric
  generation remains bounded and reproducible.

- Reference-collapse recipes use `reference_alignment_summary` with
  config-declared `reference_sets` when the analysis needs named landmark or
  standard collections. The builder emits group size, median pairwise cosine
  distance, distance IQR, and explicit `reference_set_status` fields such as
  `ok`, `absent`, `missing_rows`, `too_small`, or `missing_columns` instead of
  silently dropping absent collections. `reference_group_columns` remains
  available for broad metadata audits, but named collections should live in
  `reference_sets` so selectors, labels, and notebook overlays share one
  contract.

## Plots and notebooks

- `defaults.plot_formats` controls which image files `plot render` writes; the current renderer supports `svg`, `pdf`, and `png`.

- `plot render` supports `projection_scatter`, `projection_grid`, `heatmap`, `distance_scatter`, `xy_scatter`, `distribution`, `curve`, `correspondence_heatmap`, and `agreement_summary`.

- `plot render` accepts either a named `plots.<id>` recipe with no inline plot flags, or an ad hoc inline spec; mixing the two modes is rejected.

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
  larger reference collections can be represented without dense static
  labels.

- `notebook generate` emits one read-only workspace notebook app per
  declared `notebooks.<id>`. An artifact-group dropdown selects the saved analysis to review. Controls
  are data-driven from `controls.json`: persisted plot artifact, candidate-set
  or grid/single mode, model, representation family, context, view, projection,
  hue, and reference overlay. Plot and geometry accordions render in the same
  selected surface for the selected artifact group.
  `notebook smoke` runs both import smoke and `marimo check` so generated
  notebooks must remain valid marimo DAGs.

- Notebook controls expose both `reference_labels` for small static callout
  sets and `reference_sets` for selector-backed all/collection/subset modes.
  Reference sets may use `where` for OR-style selector membership and
  `where_all` for AND-style membership. Selectors support exact `equals`,
  `in_values`, `regex`, and `not_regex` checks so representation-specific
  suffixes for selected regions or reverse-complement rows can be included
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
  diagnostic representations can stay visible without treating missing results as evidence. Fixed-grid notebook presets include only views with at
  least one usable projection artifact by default. Set
  `notebooks.<id>.show_missing_projection_placeholders: true` only when a
  workspace intentionally wants blank missing-projection panels. The
  `candidate_inventory` and candidate-set metadata still list materialized
  output-layer or other non-projected representations for health and alignment
  review.

- Notebook controls also publish the shared `candidate_inventory` ledger used
  by workspace snapshots and catalogs. The generated Marimo runtime reads row
  counts and dimensionality from this saved inventory first, then from
  explicit geometry-control rows when the inventory has not been embedded,
  instead of opening view matrices during notebook startup.

- Inventory builders read materialized view shape from view manifest
  `stats.rows` and `stats.dims`. Current generated workspaces must surface
  missing shape stats as unknown inventory values rather than inspecting
  matrix files during notebook bootstrap.

- `notebook generate` may return `attention` when the notebook is written before the default deliverable plot exists; the explicit degraded state is persisted and `notebook smoke` remains the gate.

- `notebook generate` refuses to overwrite or regenerate a notebook when the default deliverable has freshness drift; refresh the deliverable or its linked recipe first so the notebook remains a review surface over fresh artifacts.

## Validation and artifact reuse

- `workspace refresh` clears only workspace-local LatentDNA outputs; it never mutates upstream `seqledger` datasets.

- When a materialized view artifact was built from a different source/vector
  declaration, deep validation reports `materialized_contract_status:
  stale_source_contract` instead of silently accepting the old rows as current.
  Refresh the view artifact before using downstream plots as current results.

- Deliverable loading rejects declared outputs that the linked recipe does not actually produce, including config-backed outputs such as `views`, `scalars`, `reducers`, `reduced_views`, and `exports`.

- Deliverables declare `title`, `summary`, `question`, and `section` explicitly;
  `description` and `kind` do not populate missing semantic fields.

- Deliverable status and run inventory use recorded input and source digests where available, including export and alignment manifests with explicit path-backed provenance.

## Explicit analysis tasks

`representation_scorecard` candidates declare `classification_tasks`. Each task
supplies exactly one `source_scalar` or `source_view`, a `label_column`,
`positive_values`, `score_columns`, and a `metrics` mapping from `auroc` or
`auprc` to workspace metric IDs. With multiple score columns, the row-wise
minimum is the joint score. Unknown parameters, empty candidate lists and candidates
without a computation are rejected before output is written. Each candidate must
produce at least one metric. A shared source table is read once per candidate.
Optional `reference_neighbor_tasks` declare label and reference columns and
values. `neighbor_label_enrichments` declare the label columns to compare.

`reference_alignment_summary` accepts `contrasts` with a `cohort_column`,
`target`, `background`, `reference_column`, `reference_values`, and `metric_id`.
Each contrast reports the reference direction's cosine alignment with the
target centroid minus its alignment with the background centroid. Workspace
`metric_definitions` supply the interpretation and units of named metrics.

`similarity_margin` and `cohort_similarity_margin` emit their configured
`margin_pairs`. Optional `margin_combinations` map new output column names to
an `operation` (`min` or `max`) and a nonempty list of existing `columns`.

`tf_axis_orientation_audit` requires explicit `centroid_groups`, `tf_columns`,
`tf_bin_labels`, and `axes`. Each axis maps an output column to `target` and
`background` centroid names. Optional `centroid_distances` map output columns
to centroids. An association with one group receives its configured bin label;
multiple groups produce `mixed`, and no groups produce `neither`.
`tf_axis_orientation_tests` requires explicit axis/target/background `tests`.
The `output_filter` and `where` fields declare the audited and tested populations.

## Plot presentation

Plot IDs identify artifacts. Layout comes from the plot configuration:
`single_row_panels`, `square_panels`, `panel_width`, `panel_height`,
`extra_width_per_column` and `tight_layout_*` control panel geometry.
`bar_orientation` selects horizontal or vertical metric bars. `compact_layout`
reduces title and heatmap spacing and abbreviates context labels.
`legend_single_row`, `legend_anchor_y`, `legend_base_margin` and
`legend_location` control legends. Renaming a plot does not select a different
layout.

Model identity is supplied in view tags or scorecard metadata. A candidate's
`candidate_model`, `candidate_family`, `candidate_scope` and `candidate_label`
are optional explicit descriptors and are retained in metric-pair and bar tables.
Absent descriptors leave the candidate ID as its label; IDs are not parsed to
infer a model family or size.
