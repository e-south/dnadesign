# Trait Axis Projection Dev Spec

**Owner:** dnadesign-maintainers
**Status:** implemented generic runtime contract
**Last verified:** 2026-07-14
**Last edited:** 2026-07-14
**First dogfood workspace:** `rt_lnrna_sponging_construct_triage`

This note defines a generic LatentDNA fitted-geometry primitive,
`trait_axis_projection`. The first dogfood use case is RT-lnRNA abundance
geometry, but the primitive is not RT-specific.

## Source Basis and Document Role

This dev spec is the durable LatentDNA maintainer contract for the implemented
primitive. LatentDNA owns fitted-geometry mechanics and compact provenance only
when LatentDNA fits geometry; the RT-lnRNA study owns biological meaning,
source labels, abundance overlays, GenBank/compiler/DMS categories, and future
SPOP interpretation.

Earlier design notes proposed broader population ledgers, duplicate collapse,
composition baselines, and UMAP-heavy interpretation. Those are not the first
implementation target. The implementation target is a narrower generic
trait-axis scalar primitive with explicit fit/eval/reference/sensitivity
population roles, endpoint sensitivity, row-level projections, and compact
summary sidecars.

This file is not the RT study narrative. The RT companion document should live
at:

```text
docs/studies/rt_lnrna_sponging_construct_triage/contexts/latentdna/trait-axis-projection.md
```

That companion should explain Crawford/Khan source scope, GenBank/compiler
overlay interpretation, DMS held-out status, and future Reader SPOP alignment
in RT study terms.

Audience:

- LatentDNA maintainers implementing the generic scalar primitive.
- RT-lnRNA maintainers configuring and interpreting the first dogfood pass.
- Reviewers checking that the implementation stays generic and that generated
  artifacts are produced through official commands.

Success criteria for this document:

- a Codex implementer can identify the right module, registry hook, tests, and
  validation commands without reopening the whole design discussion;
- RT-specific nouns are clearly confined to config and study-facing docs;
- first-pass deliverables and deferred work are separated;
- failure modes are explicit enough to prevent an attractive but unsupported
  biological claim.

This document defines the runtime contract; it is not evidence that a generated
workspace artifact is current. The scalar builders, RT workspace configuration,
plot semantics, and tests exist in the checked-in source tree. Use manifests,
deliverable status, and workspace validation to establish whether a particular
generated scalar, plot, or notebook is present and fresh.

## Executive Summary

The RT-lnRNA study needs a focused analysis path for one biological question:

> Does Evo2 / LatentDNA geometry encode a source-scoped multi-copy ssDNA /
> RT-DNA abundance direction, and can that direction help audit or triage
> unlabeled RT-lnRNA candidates before SPOP labels are complete?

The correct LatentDNA contribution is a generic fitted-geometry primitive. The
primitive fits a signed trait axis from declared high/low endpoint cohorts in a
representation view, then projects configured populations onto that axis. The
same machinery must support Crawford abundance, Khan abundance, future Reader
SPOP, and non-RT trait layers without adding study-specific LatentDNA runtime
code.

The first dogfood path is RT-lnRNA:

- Crawford is the dense primary source-scoped Eco1 msDNA abundance evidence.
- Khan is a sparse parallel source-scoped RT-DNA abundance audit.
- GenBank and compiler MSD rows are unlabeled reference overlays.
- RT-CDS DMS rows are sensitivity or perturbation overlays, not fit rows.
- Reader SPOP remains future label evidence and must not be invented.

The notebook and docs should present this as a staged decision funnel:

1. establish whether a source-scoped abundance axis exists;
2. compare Crawford and Khan fitted directions without pooling their numeric
   values;
3. rank representation views by trait-axis preservation;
4. project GenBank and compiler candidates;
5. later compare against Reader SPOP when materialized;
6. analyze DMS parent-relative movement only when parent mapping is reliable.

## Question

The RT-lnRNA study needs to ask:

> Does Evo2 / LatentDNA geometry encode a source-scoped multi-copy ssDNA /
> RT-DNA abundance direction, and can that direction help audit or triage
> unlabeled RT-lnRNA candidates before SPOP labels are complete?

The reusable LatentDNA question is narrower:

> Given a representation view and declared high/low endpoint cohorts for a
> trait, can LatentDNA fit a signed geometry axis and project configured
> populations onto that axis with auditable provenance?

LatentDNA owns the mechanics. The workspace and study own the biological
meaning of traits, cohorts, labels, and downstream claims.

## Boundary

LatentDNA owns:

- representation row alignment;
- L2 normalization;
- endpoint centroid fitting;
- signed axis-vector construction;
- row-level projection scores;
- summary metrics and failure status;
- compact fitted-geometry provenance;
- plot-ready scalar sidecars.

Study or workspace config owns:

- trait meaning;
- fit/eval/reference/sensitivity population selectors;
- source-value columns;
- endpoint thresholds;
- parent mapping;
- visible labels and biological interpretation.

For RT-lnRNA, this means LatentDNA must not hard-code Crawford, Khan, Eco1,
MSD, RT, lnRNA, GenBank, DMS, or SPOP. Those nouns belong in the RT workspace
config and study docs.

## Design Principles and Scope Tags

Use the following tags when implementing, reviewing, or updating the spec:

- **Current context fact:** observed or supplied study/workspace state, such as
  row counts, view names, source meanings, and existing scalar/plot surfaces.
  These facts should be revalidated against the live checkout before generated
  artifacts or scientific conclusions are refreshed.
- **Generic primitive contract:** reusable LatentDNA behavior that belongs in
  runtime code, tests, and scalar manifests.
- **RT dogfood configuration:** the first workspace use of the generic
  primitive. It may use RT nouns in YAML, plot semantics, and study docs, but
  not in generic runtime branches.
- **Interpretive claim:** a study-owned conclusion drawn from summary metrics
  and plots. The generic primitive should emit enough evidence for the claim,
  but should not encode RT biological meaning.
- **Future extension:** useful work that must not block the first pass unless a
  stable upstream field or existing LatentDNA utility makes it cheap and
  generic.

Prefer explicit contract errors. Do not add alias paths, silent fallbacks,
broad source selectors, or hidden row collapsing to make the first RT dogfood
pass look cleaner.

## Goals

The implemented first pass supports:

- a generic `trait_axis_projection` primitive for fitted high/low endpoint
  geometry;
- Crawford and Khan axes configured separately in the RT workspace;
- multi-view ranking by trait-axis preservation, not by source-family
  separation as a positive objective;
- row-level projection of fit, eval, reference, and sensitivity populations;
- GenBank and compiler MSD reference overlays;
- held-out RT-CDS DMS overlay with DMS excluded from abundance-axis fitting by
  default;
- optional parent-relative DMS movement fields when parent mapping is complete:
  `axis_delta` and `orthogonal_delta`;
- endpoint sensitivity across min/max, top/bottom 5 percent, top/bottom
  10 percent, and top/bottom 20 percent;
- compact summary metrics suitable for scorecards, notebooks, and future
  plot-semantic surfaces;
- a clean future path for Reader-owned SPOP labels as another trait layer.

## Non-Goals

- Do not add an RT-specific LatentDNA runtime module.
- Do not pool Crawford and Khan abundance into one numeric target.
- Do not treat abundance overlays as OPAL `Y`.
- Do not infer or invent SPOP labels.
- Do not use UMAP as primary evidence for abundance geometry.
- Do not let perturbation/sensitivity rows define an axis by default.
- Do not collapse meaningful near-duplicates or single-nucleotide variants.
- Do not block the first pass on broad GC, k-mer, structure, or edit-distance
  baselines.
- Do not hand-edit generated outputs.

Exact accidental duplicates still need explicit handling. A repeated construct
subject ID or exact repeated sequence/source observation should fail or be
made explicit by config. Near-duplicates are not duplicate hygiene.

## Current RT-lnRNA Context

The first dogfood workspace is
`rt_lnrna_sponging_construct_triage`. The current checked-in study context
describes:

- 10,415 Construct subjects;
- 20,830 realized 2,000 bp context rows;
- 62,490 sequence-view rows;
- source split:
  - 36 GenBank catalog subjects;
  - 4,148 Crawford subjects;
  - 71 Khan subjects;
  - 80 compiler MSD subjects;
  - 6,080 RT-CDS DMS subjects.

One Construct subject represents a paired lnRNA cassette plus RT CDS cassette
inside a fixed 2,000 bp dual-cassette context. The biological identity is
`construct_subject__id`, with biological sequence authority in
`construct_subject__lnrna_sequence` and `construct_subject__rt_cds_sequence`.
The carrier row sequence is not the biological identity.

Infer consumes six source views:

1. full 2,000 bp forward context;
2. full 2,000 bp reverse-complement context;
3. lnRNA-centered 384 bp forward pooling window;
4. lnRNA-centered 384 bp reverse-complement pooling window;
5. RT CDS-centered 1,600 bp forward pooling window;
6. RT CDS-centered 1,600 bp reverse-complement pooling window.

LatentDNA derives bidirectional concat views and lnRNA + RT CDS pair views for
review and candidate-X selection after sidecars exist. The trait-axis primitive
should fit and score declared LatentDNA views; it should not recreate Construct
or Infer source-view logic.

Current source semantics:

- Crawford rows are Eco1-local lnRNA/MSD variant-library rows with continuous
  msDNA abundance values. They are dense enough for the primary continuous
  source-value analysis.
- Khan rows are sparse cross-retron RT-lnRNA rows with source-scoped RT-DNA
  abundance values. They must not be numerically pooled with Crawford.
- GenBank rows are sequence-authority/reference rows, not abundance fit rows by
  default.
- Compiler MSD rows are study-owned design references and unlabeled for
  abundance/SPOP in the first pass.
- RT-CDS DMS rows are in silico RT CDS point-mutation variants and unlabeled
  sensitivity rows by default.
- Reader SPOP is future Reader-owned sponging-function label evidence, not
  currently materialized as a LatentDNA trait.

The implementation should re-run workspace validation before relying on these
counts or column names in a commit message or generated artifact. The counts are
the source basis for this spec; they are not a substitute for current validation
at implementation time.

## Placement

Implemented runtime module:

```text
src/dnadesign/latentdna/src/scalars/builders/trait_axis_projection.py
```

Thin registry hook:

```text
src/dnadesign/latentdna/src/scalars/build.py
```

Generic maintainer doc:

```text
src/dnadesign/latentdna/docs/dev/trait-axis-projection.md
```

RT study-facing companion:

```text
docs/studies/rt_lnrna_sponging_construct_triage/contexts/latentdna/trait-axis-projection.md
```

RT workspace configuration:

```text
src/dnadesign/latentdna/workspaces/rt_lnrna_sponging_construct_triage/config.yaml
```

If workspace config composition exists before implementation, prefer a
deterministic resolved fragment over bloating the main YAML file.

## Builder Shape

Prefer two builders.

### `trait_axis_projection_rows`

Fits configured axes from explicit endpoint cohorts, scores configured
populations, and emits row-level sidecars.

Required behavior:

- resolve candidate views;
- resolve fit and score populations;
- select high/low endpoint rows;
- fit centroids per view/axis/endpoint definition;
- score all configured populations;
- emit compact fit/scored provenance;
- write row-level Parquet sidecars.

### `trait_axis_projection_summary`

Consumes row-level projections and emits compact scorecard-friendly summaries.

Required behavior:

- compute row counts by population role and endpoint definition;
- compute source-value correlations where finite values exist;
- compute endpoint effect size;
- summarize endpoint-sensitivity stability;
- carry degenerate or invalid status with reasons;
- optionally compute axis-vector concordance between comparable axes.

A single builder is acceptable only if existing scalar patterns strongly favor
it and row-level and summary outputs remain independently testable.

## Current Scalar Runtime Integration Notes

Current preassay scalar builders are registered through
`src/dnadesign/latentdna/src/scalars/preassay.py` and usually return a simple
`(table, inputs, stats)` tuple that `build_preassay_scalar_artifact` writes to
`table.parquet`. `trait_axis_projection_rows` is more demanding because it must
declare fitted-geometry provenance and, if axis concordance is supported, a
compact fitted-axis sidecar.

`trait_axis_projection_rows` and `trait_axis_projection_summary` use the broad
scalar `build.py` dispatch path and return full `BuiltScalarArtifact` objects.
This keeps `provenance.json`, `fitted_axes.parquet`, and
`axis_concordance.parquet` declared in scalar manifests. Do not write
undeclared files under `outputs/scalars/<scalar_id>/`; every Parquet, JSON, or
NPZ sidecar must appear in the manifest so freshness, deliverables, and
downstream summary builders can reason about it.

The implementation should reuse existing utilities where they match the
contract:

- `ScalarInputRef` and `BuiltScalarArtifact` for artifact manifests;
- `_require_param` and `_optional_param` for current parameter conventions, or
  a typed params model if scalar registries are upgraded first;
- `_load_view_scope_table` or the current scope resolver for aligned view
  rows/matrices;
- `pearson_correlation`, `spearman_correlation`, and `kendall_tau_b` from
  `src/dnadesign/latentdna/src/stats/rank.py`;
- `l2_normalize_vector` for endpoint centroids and axis vectors.

Be careful with normalization helpers. Existing preassay helpers commonly use
`standardize_and_l2_normalize`, which centers and scales before row
normalization. This primitive's math contract is row L2 normalization of the
declared representation vectors. If maintainers intentionally adopt
standardize-then-L2 for consistency with existing review scalars, that must be
an explicit config field and provenance value, not an accidental helper reuse.

### Adjacent Primitive Reuse

Use nearby scalar code for mechanics, not for biological meaning:

- `ordinal_axes_audit` and `ordinal_ladder_rows` are useful references for
  source-scoped value/bin handling and existing RT abundance surfaces, but they
  do not answer out-of-sample projection of reference and sensitivity
  populations.
- `reference_to_centroid_similarity` is useful for cohort centroid mechanics,
  but trait-axis projection needs a signed high-minus-low direction and
  continuous source-value summaries.
- `axis_centroid_distance` is useful for centroid-distance diagnostics, but the
  first-pass evidence path is row-level projection and trait preservation across
  views.
- `tf_axis_orientation` may provide useful `BuiltScalarArtifact` and sidecar
  patterns. Do not copy TF, promoter, or association-overlay semantics into the
  generic trait-axis primitive.

The new module should be cohesive rather than a large expansion of
`preassay_selection.py`. The registry hook should be thin enough that removing
or replacing the primitive would not disturb unrelated preassay builders.

## Config Concepts

The exact YAML shape should follow existing workspace conventions, but it must
preserve these concepts:

```yaml
kind: trait_axis_projection_rows
candidate_views:
  - <view_id>

axes:
  - trait_id: <generic_trait_id>
    label: <display_label>
    source_value_column: <optional_numeric_column>
    candidate_id_column: <optional_identity_column>

    fit_population:
      population_id: <id>
      role: fit
      where:
        - column: <metadata_column>
          equals: <value>

    endpoint_groups:
      method: <quantile | explicit_values | min_max | configured_selector>
      value_column: <optional_numeric_column>
      group_column: <optional_categorical_column>
      low_values: <optional_list>
      high_values: <optional_list>
      low_quantile: <optional_float>
      high_quantile: <optional_float>
      min_low_rows: <int>
      min_high_rows: <int>

    score_populations:
      - population_id: <id>
        role: <fit | eval | reference | sensitivity>
        where:
          - column: <metadata_column>
            equals: <value>

    parent_key: <optional_column>
    parent_candidate_id_column: <optional_column>
    confound_columns:
      - <optional_column>

    endpoint_sensitivity:
      enabled: true
      endpoint_definitions:
        - min_max
        - quantile_0_05
        - quantile_0_10
        - quantile_0_20
```

Allowed generic population roles:

- `fit`: rows allowed to fit endpoint centroids.
- `eval`: labeled rows scored for validation.
- `reference`: unlabeled or provenance rows projected for interpretation.
- `sensitivity`: perturbation or stress-test rows scored after fit.
- `excluded`: optional provenance status for deliberate exclusions.

The generic code should enforce role semantics, not study semantics. For
example, rows configured as `sensitivity` should not also fit the axis unless an
explicit override exists and is tested.

Selectors should support the current workspace selector vocabulary and add only
the minimal predicates needed for this primitive:

- `equals`;
- `in_values`;
- `regex` / `not_regex` where already supported by workspace code;
- `finite: true` for numeric source-value and parent/projection fields;
- optional `not_null: true` if parent mapping or source observations need it.

Selector errors should name the scalar ID, axis ID, population ID, column, and
predicate that failed. Missing columns should fail before matrix loading.

Identity columns must be explicit. The row output needs a stable
`candidate_id`; the RT workspace currently aligns major view comparisons on
`subject_key`, while biological subject identity is the Construct subject ID.
The builder should accept a `candidate_id_column` or derive it from the
workspace's established view-row identity contract. It must not infer identity
from display labels.

### Config Validation Rules

Validate config before loading large matrices whenever possible.

Required validation:

- every `trait_id`, `population_id`, `axis_id`, and
  `endpoint_definition_id` is unique within its declared scope;
- `candidate_views` is non-empty and each view exists in the resolved
  workspace;
- each selector column exists in the aligned view-row metadata or declared
  scalar input before matrix loading;
- `source_value_column` exists when configured and is numeric or coercible by
  an explicit numeric derivation contract;
- each score population has one allowed generic role;
- a `fit_population` role is exactly `fit`;
- `sensitivity` populations cannot overlap fit rows unless a deliberate,
  named override is configured and tested;
- `parent_key` and `parent_candidate_id_column` are present together when
  parent-relative outputs are requested;
- every endpoint definition has enough information to select both low and high
  rows.

Selector semantics should be deterministic. If an implementation supports both
workspace-local selector helpers and new trait-axis-only predicates, document
the merge order and reject ambiguous predicates rather than guessing.

Avoid adding a new global selector language in this pass. Extend existing
workspace selector conventions only where the primitive cannot otherwise
express finite numeric source values, parent mapping, or endpoint membership.

## Endpoint Selection Contract

Endpoint selection is a generic operation over metadata and optional numeric
source values. It must not know about Crawford, Khan, SPOP, or RT categories.

Supported first-pass endpoint methods should include:

- `min_max`: one or more rows with the minimum finite value become low
  endpoints, and one or more rows with the maximum finite value become high
  endpoints;
- `quantile`: low endpoints are the bottom configured fraction and high
  endpoints are the top configured fraction;
- `explicit_values`: low/high endpoints come from configured categorical values
  in a group column;
- `configured_selector`: low/high endpoints are separate selector blocks if
  existing workspace selector helpers make that cleaner.

Quantile selection must be deterministic for ties. If ties at a threshold would
make endpoint membership ambiguous, the builder should either include all tied
rows with explicit provenance or fail with a clear validation error. Do not
silently drop tied endpoint rows.

Endpoint sensitivity should be implemented as multiple endpoint definitions for
the same trait and view:

- `min_max`;
- `quantile_0_05`;
- `quantile_0_10`;
- `quantile_0_20`.

Each endpoint definition should produce its own row-level scores and summary
rows, or an equivalent normalized output where `endpoint_definition_id` is
present on every row. The first pass may plot only top-ranked views or appendix
views, but the summary table should include every configured endpoint
definition.

## Math

Use L2-normalized representation rows.

```text
z = x / ||x||_2
```

Rows with zero or non-finite norm must be excluded or marked invalid with
explicit provenance.

For each configured axis:

```text
c_high_raw = mean(z_i for i in high endpoint rows)
c_low_raw  = mean(z_i for i in low endpoint rows)

c_high = c_high_raw / ||c_high_raw||_2
c_low  = c_low_raw  / ||c_low_raw||_2
```

Degenerate endpoint centroids invalidate that axis/view/endpoint definition.

For each scored row:

```text
similarity_to_high = dot(z, c_high)
similarity_to_low  = dot(z, c_low)
endpoint_margin    = similarity_to_high - similarity_to_low
axis_vector        = normalize(c_high - c_low)
axis_projection    = dot(z, axis_vector)
```

Emit both `endpoint_margin` and `axis_projection`. They are related but not the
same review surface.

Parent-relative movement, when configured:

```text
axis_delta = mutant_projection - parent_projection

parent_residual = z_parent - parent_projection * axis_vector
mutant_residual = z_mutant - mutant_projection * axis_vector
orthogonal_delta = ||mutant_residual - parent_residual||_2
```

This is representation movement, not a predicted functional effect.

Axis-vector concordance, when comparing two fitted axes in one view:

```text
axis_concordance = dot(axis_vector_a, axis_vector_b)
axis_angle = arccos(clamp(axis_concordance, -1, 1))
```

Score-score scatter and direct axis-vector concordance answer different
questions and should not be collapsed.

## Row Builder Algorithm

The row builder should follow this order so failures are early and matrix work
is not repeated.

1. Parse and validate params.
   - Validate required keys, allowed roles, selector predicates, endpoint
     definitions, minimum row counts, and source/parent columns.
   - Expand endpoint sensitivity definitions into normalized
     `endpoint_definition_id` entries.
2. Resolve all candidate views.
   - Confirm each view exists and exposes a matrix plus row table.
   - Confirm all required metadata columns exist in the view rows before
     loading the matrix where the view-row contract allows that.
3. For each view, load the matrix once.
   - Align matrix rows and metadata rows.
   - Apply the declared normalization policy once.
   - Build vectorized boolean masks for each fit and score population.
4. For each axis and endpoint definition, fit centroids.
   - Select fit rows.
   - Select low/high endpoints inside the fit population.
   - Validate non-empty, non-overlapping, minimum counts, and non-degenerate
     centroids.
   - Orient the axis from low toward high.
5. Score populations in chunks.
   - Compute `similarity_to_high`, `similarity_to_low`, `endpoint_margin`, and
     `axis_projection`.
   - Attach source values only where configured and finite.
   - Attach endpoint membership for fit rows.
   - Attach parent-relative fields only when configured and resolvable.
6. Write row table and declared sidecars.
   - Emit row-level projections as Parquet.
   - Emit compact provenance and any declared fitted-axis sidecar.
   - Record invalid/skipped row and axis counts in stats.

Use NumPy arrays for masks and score columns. Avoid a large Python
list-of-dicts loop for every scored row when the projected table can be built
from arrays and Arrow/Pandas/Polars columns.

## Row Output

`trait_axis_projection_rows` should emit one row per scored candidate/view/axis
and population membership.

Minimum columns:

- `candidate_id`
- `candidate_id_column`
- `view_id`
- `trait_id`
- `axis_id`
- `endpoint_definition_id`
- `population_id`
- `population_role`
- `similarity_to_high`
- `similarity_to_low`
- `endpoint_margin`
- `axis_projection`
- `endpoint_group` for fit rows where applicable
- `source_value` when configured and finite
- `source_value_column` when configured
- `source_value_available`
- `row_status`
- `row_status_reason`

Recommended columns:

- workspace or view-row identity fields needed for joins, such as
  `subject_key` or Construct subject ID when present;
- endpoint membership flags, such as `is_low_endpoint` and `is_high_endpoint`;
- fit/scoring grain fields when a view can contain multiple rows per subject;
- continuous source-value rank within the fit population when helpful for
  endpoint-sensitivity plots;
- parent-relative fields when configured:
  - `parent_candidate_id`
  - `parent_key`
  - `parent_axis_projection`
  - `axis_delta`
  - `orthogonal_delta`

The row table should carry enough metadata for plot labels and joins without
opening raw matrix payloads.

Population membership can be many-to-one. A scored row may appear once per
population membership if selectors intentionally overlap, but overlap between
`fit` and `sensitivity` must be rejected unless explicitly allowed. The row
output should make membership explicit rather than collapsing roles into a
single display label.

## Summary Output

`trait_axis_projection_summary` should emit one row per `trait_id`, `axis_id`,
`view_id`, and endpoint definition.

Minimum fields:

- total scored row count;
- fit population row count;
- low endpoint row count;
- high endpoint row count;
- eval/reference/sensitivity row counts;
- source-value Spearman and Pearson where finite values exist;
- Kendall if available and practical;
- endpoint effect size;
- endpoint-sensitivity stability summary;
- degenerate or failure status and reason.

Future or optional fields:

- bootstrap confidence interval;
- null/permutation score;
- direct axis-vector concordance;
- axis angle;
- claim classification such as `supported`, `source_confounded`,
  `view_fragile`, `not_supported`, or `insufficient`.

First pass may defer expensive null/permutation or bootstrap surfaces, but the
omission must be documented.

### Summary Metric Definitions

Use deterministic, simple definitions first.

Correlation metrics:

- compute source-value correlations only over rows with finite `source_value`
  and finite score values;
- report the row scope used for each correlation, for example
  `fit_population`, `eval_population`, or `all_labeled_scored`;
- prefer `axis_projection` and also emit `endpoint_margin` correlations when
  practical;
- emit `NaN` plus an explicit invalid reason when finite pair counts are below
  configured minima.

Endpoint effect size:

- compute the difference between high-endpoint and low-endpoint score means;
- include a standardized effect size such as pooled-standard-deviation
  Cohen-style `d` when both endpoint score groups are non-degenerate;
- record the score column used: `axis_projection` or `endpoint_margin`.

Endpoint-sensitivity stability:

- compare the primary endpoint definition against each sensitivity definition
  for the same trait/view;
- report correlation sign stability, effect-size sign stability, and score-rank
  stability where finite scored rows overlap;
- if axis vectors are available, report axis-vector cosine to the primary
  endpoint definition;
- treat a sign flip as meaningful unless the implementation has a documented
  orientation rule that realigns equivalent axes.

Implementation choice as of 2026-05-27: the summary builder consumes the
row-builder-declared `fitted_axes.parquet` sidecar when present, declares that
sidecar as a summary input, and emits `axis_vector_primary_concordance` plus
`axis_vector_primary_angle` for endpoint-sensitivity rows. This keeps notebook
and scorecard consumers on compact scalar sidecars rather than raw matrices.
Rows with a requested but incomplete parent mapping are preserved as scored rows
with `row_status=invalid`, an explicit `row_status_reason`, and null
parent-relative deltas; summary rows include invalid-row counts and compact
reason counts so downstream panels can report partial parent-map failures.

Axis-vector concordance:

- only compare axes fitted in the same `view_id`, dimensionality, and
  normalization policy;
- compare Crawford and Khan as independently fitted directions, not pooled
  numeric labels;
- emit both cosine concordance and angle when vectors are available;
- keep score-score scatter as a separate row-level or plot surface.

## Summary Builder Algorithm

The summary builder should not reload raw representation matrices.

1. Load the row-level projection scalar table and manifest.
2. Validate that the source scalar was produced by
   `trait_axis_projection_rows` or declares the expected row-level columns.
3. Group by `trait_id`, `axis_id`, `view_id`, `endpoint_definition_id`, and
   score column where needed.
4. Compute counts and invalid/skipped reasons.
5. Compute source-value correlations for each configured correlation scope.
6. Compute endpoint effect-size metrics.
7. Compute endpoint-sensitivity stability by comparing endpoint definitions
   within trait/view.
8. Compute axis-vector concordance only from declared compact sidecars or from
   row-builder-emitted concordance rows. Do not reopen `matrix.npy`.
9. Emit summary rows in scorecard-friendly long form, plus any compact wide
   table only when existing plot kinds require it.

If the row builder emits invalid axis/view/endpoint records, the summary builder
must preserve those records with `status` and `reason` rather than silently
dropping them. This is how the notebook can distinguish `not_supported` from
`insufficient`.

## Provenance

Because this primitive fits geometry, LatentDNA must emit compact
fitted-geometry provenance. This is not a requirement for every LatentDNA plot.

Provenance should include:

- `trait_id`;
- `axis_id`;
- `view_id`;
- endpoint definition;
- compact fit selector or selector digest;
- score population IDs and roles;
- fit row counts;
- endpoint row counts;
- failure/warning status;
- source-value column name if configured;
- parent-key column name if configured.

The notebook can show compact counts. Full audit detail belongs in the scalar
sidecars and provenance output.

## Artifact Contract

Follow existing scalar artifact conventions. The exact manifest schema should
match the current LatentDNA scalar builder contract, but the implementation
must preserve these logical outputs.

### Row Builder Outputs

`trait_axis_projection_rows` should write:

- primary scalar table: row-level projections, preferably
  `outputs/scalars/<scalar_id>/table.parquet` if that is the local scalar
  convention;
- fitted-axis provenance sidecar, for example
  `outputs/scalars/<scalar_id>/provenance.json`;
- optional compact endpoint membership sidecar when endpoint labels would make
  the projection table too wide;
- optional fitted-axis sidecar, for example `axes.npz` or
  `axes.parquet`, containing axis vectors only when needed for direct
  concordance or parent-relative scoring;
- optional axis-vector digest or compact metadata. Do not require notebook
  controls to load high-dimensional axis vectors.

The scalar manifest should declare every sidecar it writes and every upstream
input it used:

- view matrix or view artifact IDs;
- view rows or metadata source IDs;
- source scalar IDs if source values are read from a scalar sidecar;
- parent mapping or mutation annotation sources when used.

The manifest `stats` block should include at least:

- configured trait count;
- configured view count;
- scored row count;
- invalid/skipped row count;
- failed axis/view/endpoint-definition count;
- wall-time and peak-memory fields if the scalar build framework already
  exposes them.

Axis-vector sidecars are implementation artifacts, not notebook controls. They
may be large enough to matter for concat views. If direct concordance or
parent-relative scoring can be computed inside the row builder and summarized
without persisting vectors, prefer compact JSON/provenance plus summary metrics.
If a downstream summary builder needs vectors, persist them in a declared
sidecar and keep plot/notebook consumers on summaries.

### Summary Builder Outputs

`trait_axis_projection_summary` should write:

- primary summary table, preferably
  `outputs/scalars/<summary_scalar_id>/table.parquet`;
- any compact side tables needed by scorecard or plot surfaces, with manifest
  declarations;
- summary provenance that points back to the row-level scalar ID and row-level
  manifest.

The summary builder should consume row-level projections, not reload raw
representation matrices. Axis-vector concordance is the exception: if direct
axis-vector concordance is implemented in the summary builder, the row builder
must emit the needed compact axis-vector artifact or concordance-ready
sidecar.

### Stable Identifiers

Use stable IDs that are reusable across studies:

- `trait_id` identifies the configured trait layer.
- `axis_id` identifies a fitted axis for a trait/view/endpoint definition.
- `endpoint_definition_id` identifies min/max or quantile sensitivity choices.
- `population_id` identifies a configured score population.
- `population_role` is one of the generic roles.

Do not derive biological meaning from ID substrings in generic code. IDs may be
descriptive for humans, but all behavior must come from config fields.

## Failure Conditions

Validate before expensive matrix work where possible.

The builder should fail fast or emit explicit invalid summaries when:

- an unknown trait, view, endpoint definition, population role, or selector
  predicate appears in config;
- endpoint selectors match no rows;
- high and low endpoint rows overlap;
- expected endpoint minimum row counts are not met;
- endpoint centroids are degenerate;
- row normalization produces zero or non-finite vectors under the configured
  normalization policy;
- fit rows and view matrices cannot be aligned;
- score populations cannot be resolved;
- configured `source_value_column` is missing;
- finite source values are required but absent;
- a sensitivity role overlaps fit without explicit override;
- parent-relative outputs are requested but parent mapping is missing or
  ambiguous;
- metadata row count and matrix row count disagree.

Invalid summaries are acceptable for per-axis/per-view degeneracy after config
resolution, but config errors and missing required columns should fail the
scalar build before expensive matrix work.

## Weighting and Duplicate Policy

Default first-pass fitting is row-weighted. Do not collapse Crawford
near-duplicates, single-nucleotide variants, or intentionally repeated design
neighborhoods. In this study those near-duplicates are biological signal, not
noise.

The only required duplicate hygiene is exact accidental duplicate handling:

- the same `candidate_id` or construct-subject key should not appear twice in a
  fitted view unless the row grain makes that repetition explicit;
- exact same sequence plus exact same source observation should either be
  rejected or carried through with an explicit configured rule;
- exact same sequence with different source observations should require a
  documented row-grain policy before it can fit an endpoint;
- near-duplicate or edit-neighbor sequences remain first-class rows.

Subject-weighted or source-observation-weighted fitting may be added later, but
it must be explicit in config and recorded in provenance. Do not add automatic
deduplication as a quiet convenience.

## RT-lnRNA Dogfood

Configure Crawford and Khan as separate trait axes.

### Crawford

- Example `trait_id`: `crawford_eco1_msdna_abundance`.
- Source value: continuous Crawford msDNA abundance.
- Fit population: Crawford rows with finite abundance values.
- Endpoint selection: source-scoped high/low abundance endpoints.
- Sensitivity: min/max and top/bottom 5, 10, and 20 percent.
- Primary evidence: continuous abundance vs `axis_projection` and
  `endpoint_margin`.
- Do not collapse near-duplicates.

### Khan

- Example `trait_id`: `khan_rt_dna_abundance`.
- Source value: Khan RT-DNA abundance.
- Fit population: Khan rows with finite source-scoped abundance.
- Endpoint selection: quantile or explicit high/low groups depending on row
  count and metadata.
- Interpretation: sparse independent audit, not a pooled Crawford/Khan target.

Score populations for each axis:

- Crawford labeled rows as `fit` or `eval`, depending on split policy.
- Khan labeled rows as `eval` or parallel labeled source.
- GenBank references as `reference`.
- Compiler MSD candidates as `reference`.
- RT-CDS DMS rows as `sensitivity`.

DMS must not enter the fit population by default.

Candidate views should include the decision-funnel views:

- full 2,000 bp construct view;
- lnRNA span view;
- RT CDS span view;
- lnRNA + RT CDS pair view.

Forward/reverse-complement and bidirectional concat variants should be included
only where already part of the candidate-X review surface or needed for context
robustness.

### Example RT Workspace Shape

Use current workspace column names where they are already stable. The exact
YAML should follow the implementation's typed config model, but the resolved
config should express this shape:

```yaml
kind: trait_axis_projection_rows
candidate_id_column: subject_key
candidate_views:
  - intermediate_embedding_7b_dual_cassette_2000bp_fwd_rc_concat
  - intermediate_embedding_7b_lnrna_fixed_384bp_window_in_construct_anchor_mean_bidir_concat
  - intermediate_embedding_7b_rt_cds_fixed_1600bp_window_in_construct_anchor_mean_bidir_concat
  - intermediate_embedding_7b_lnrna_384bp_rt_cds_1600bp_anchor_window_pair_concat

axes:
  - trait_id: crawford_eco1_msdna_abundance
    label: Crawford Eco1 msDNA abundance
    source_value_column: crawford_abundance_raw_value
    primary_endpoint_definition_id: quantile_0_10
    fit_population:
      population_id: crawford_abundance_fit
      role: fit
      where:
        - {column: source_family, equals: crawford_eco1_lnrna_fixed_wt_rt}
        - {column: crawford_abundance_raw_value, finite: true}
    endpoint_groups:
      method: quantile
      value_column: crawford_abundance_raw_value
      low_quantile: 0.10
      high_quantile: 0.90
      min_low_rows: 10
      min_high_rows: 10
    endpoint_sensitivity:
      enabled: true
      endpoint_definitions: [min_max, quantile_0_05, quantile_0_10, quantile_0_20]
    score_populations:
      - population_id: crawford_labeled
        role: fit
        where:
          - {column: source_family, equals: crawford_eco1_lnrna_fixed_wt_rt}
      - population_id: khan_labeled
        role: eval
        where:
          - {column: source_family, equals: khan_abundance_affiliated_rt_lnrna_reference}
      - population_id: genbank_catalog
        role: reference
        where:
          - {column: source_family, equals: genbank_variant_catalog}
      - population_id: compiler_msd
        role: reference
        where:
          - {column: source_family, equals: compiler_generated_msd_lnrna_variant}
      - population_id: rt_cds_dms
        role: sensitivity
        where:
          - {column: source_family, equals: in_silico_rt_cds_dms}

  - trait_id: khan_rt_dna_abundance
    label: Khan RT-DNA abundance
    source_value_column: khan_abundance_normalized_value
    primary_endpoint_definition_id: quantile_0_20
    fit_population:
      population_id: khan_abundance_fit
      role: fit
      where:
        - {column: source_family, equals: khan_abundance_affiliated_rt_lnrna_reference}
        - {column: khan_abundance_normalized_value, finite: true}
    endpoint_groups:
      method: quantile
      value_column: khan_abundance_normalized_value
      low_quantile: 0.20
      high_quantile: 0.80
      min_low_rows: 3
      min_high_rows: 3
```

A corresponding summary scalar should reference the row scalar rather than
repeat the matrix/view configuration:

```yaml
kind: trait_axis_projection_summary
source_scalar: rt_lnrna_trait_axis_projection_rows
correlation_scopes:
  - population_roles: [fit, eval]
    require_source_value: true
score_columns:
  - axis_projection
  - endpoint_margin
concordance:
  enabled: true
  compare_trait_ids:
    - [crawford_eco1_msdna_abundance, khan_rt_dna_abundance]
```

This example is intentionally RT-owned config. The generic runtime should see
only selectors, numeric source columns, endpoint definitions, views, and
population roles.

### RT Dogfood Deliverables

The RT configuration slice should produce or enable these review artifacts:

- a row-level projection scalar for Crawford and Khan axes across the selected
  candidate views;
- a summary scalar suitable for the view scorecard and notebook gate panels;
- a reference/compiler placement surface that can rank GenBank and compiler MSD
  rows without treating them as labels;
- an endpoint-sensitivity summary surface that includes every configured
  endpoint definition even if full plots are appendix-level;
- a Crawford-vs-Khan score scatter and, if available, direct axis-vector
  concordance summary;
- a DMS held-out score distribution, plus parent-relative deltas only if parent
  mapping is complete;
- plot semantics that make the claim boundary visible: audit/triage evidence,
  not SPOP or OPAL `Y`.

If any deliverable cannot be implemented in the first pass, record it in the RT
study-facing companion doc and in the implementation notes as deferred, with the
reason.

## Decision Funnel

The RT notebook and study docs should present this as staged evidence. Later
gates become interpretive or appendix-level if earlier gates fail.

### Gate 1: Axis Existence

Question answered:

Does a source-scoped abundance axis exist in Evo2 / LatentDNA geometry for a
configured trait and representation view?

Required inputs:

- one candidate view matrix and aligned metadata row table;
- a source-scoped numeric value column, such as Crawford msDNA abundance or
  Khan RT-DNA abundance;
- a fit population selector;
- high/low endpoint definition;
- labeled fit or eval rows with finite source values.

Outputs and plots:

- continuous source abundance vs `axis_projection`;
- continuous source abundance vs `endpoint_margin`;
- endpoint high/low score distributions;
- summary panel with Spearman, Pearson, Kendall when available, endpoint
  effect size, row counts, and endpoint-sensitivity stability;
- invalid summary rows when endpoints are too small, overlapping, or
  degenerate.

Decision rule:

- Supported when continuous source abundance correlates with axis score,
  endpoint separation is meaningful, and sensitivity checks do not flip the
  conclusion.
- Not supported when source abundance does not stratify along the axis.
- Insufficient when endpoint populations are too small, source values are
  missing, or the fitted centroids are degenerate.

Failure mode:

- If Crawford does not stratify, do not rank unlabeled candidates by Crawford
  abundance geometry.
- If Khan does not stratify, keep Khan as weak, secondary, or unsupported.
- Do not promote an attractive UMAP separation as a substitute for this gate.

### Gate 2: Crawford/Khan Concordance

Question answered:

Do independently fitted Crawford and Khan abundance directions agree
geometrically without pooling their numeric source values?

Required inputs:

- Crawford-fitted axis for a given view;
- Khan-fitted axis for the same view;
- row-level scores for a common scored population;
- compact axis-vector sidecar or row-builder-emitted concordance metrics when
  direct vector concordance is implemented.

Outputs and plots:

- Crawford-fitted axis score vs Khan-fitted axis score scatter;
- optional direct axis-vector concordance and angle;
- score-score agreement summary by view;
- hue or facet by generic `population_role` and RT-owned source category in
  plot config, not in generic runtime logic.

Decision rule:

- Positive concordance supports source-transferable abundance-like geometry.
- Orthogonal or weakly related directions suggest Crawford Eco1-local and Khan
  cross-retron abundance behave differently.
- Source-island separation is a confound diagnostic, not positive evidence.

Failure mode:

- If the axes do not agree, do not imply a universal abundance direction.
- Continue with source-specific interpretation only.
- If direct axis-vector concordance is deferred, the doc and summary output
  must say that score-score scatter is not the same geometric test.

### Gate 3: View Selection

Question answered:

Which representation view best preserves the abundance-aligned direction?

Required inputs:

- Gate 1 summaries across configured candidate views;
- Gate 2 concordance where available;
- representation health and context robustness summaries;
- dimensionality or storage-cost metadata for candidate views.

Outputs and plots:

- trait-axis view scorecard;
- view ranking table;
- endpoint-sensitivity summary by view;
- optional frontier-style plot only when existing scorecard/frontier surfaces
  can express the decision without promoting source-family separation as a
  positive objective.

Decision rule:

- Prefer the smallest clean view that preserves Crawford abundance behavior
  and does not contradict Khan or context-robustness evidence.
- Do not pick the largest concat by default.
- Treat source-family separation as a confound diagnostic.

Suggested ranking criteria:

- Crawford continuous abundance correlation;
- endpoint high/low effect size;
- bootstrap stability when implemented;
- endpoint-sensitivity stability;
- Khan independent support or Crawford/Khan concordance;
- context robustness;
- representation health;
- dimensional cost.

Failure mode:

- If only one unstable view shows signal, classify the result as
  `view_fragile`.
- If no view shows signal, the abundance-axis interpretation is
  `not_supported`.
- If the top view is biologically implausible for the trait, for example a
  Crawford signal appearing only in an RT-CDS-only view, surface that as a
  confound or leakage concern rather than a clean win.

### Gate 4: Reference and Compiler Placement

Question answered:

Where do unlabeled GenBank and compiler MSD candidates fall on the fitted
abundance-like axis?

Required inputs:

- selected or top-ranked fitted axis;
- row-level scores for reference populations;
- labeled abundance rows for context;
- row-identifiable anchors, such as retron26 or retron43, when present.

Outputs and plots:

- ranked projection table;
- distribution of `axis_projection` by `population_role` and RT-owned source
  category;
- reference/compiler strip plot, scatter, or compact table against labeled
  abundance anchors;
- exported plot labels that preserve RT terminology and distinguish audit
  placement from functional claims.

Decision rule:

- Candidates near high-abundance-like geometry may be prioritized for audit.
- Candidates near low-abundance-like geometry may be deprioritized or flagged.
- Off-axis or out-of-distribution candidates require caution.

Failure mode:

- High-abundance-like placement is not proof of function.
- If placement is dominated by source category islands or an unstable view,
  do not use the axis for candidate triage.
- If row-identifiable anchors are absent, do not invent anchor labels.

### Gate 5: Future SPOP Alignment

Question answered:

When Reader-owned SPOP labels exist, does abundance-like geometry align with
sponging function?

Required inputs:

- a frozen or explicitly versioned abundance-axis protocol;
- Reader-owned SPOP metric and label-coverage metadata;
- source/category annotations for labeled and unlabeled rows;
- row-level abundance-axis scores from the selected view or predeclared view
  ranking rule.

Outputs and plots:

- abundance-axis score vs SPOP numeric metric;
- SPOP coverage ledger or compact coverage summary;
- quadrant interpretation:
  - high abundance / high SPOP;
  - high abundance / low SPOP;
  - low abundance / high SPOP;
  - low abundance / low SPOP;
- optional SPOP trait axis fitted by the same generic primitive.

Decision rule:

- High abundance and high SPOP supports abundance-like geometry as a possible
  in silico triage proxy.
- High abundance and low SPOP means abundance is not sufficient for sponging.
- Low abundance and high SPOP marks potentially efficient or mechanistically
  distinct candidates.
- No relationship means abundance geometry should not be used as a SPOP proxy.

Failure mode:

- No SPOP labels means no SPOP claims.
- Sparse or non-random SPOP coverage must be explicit.
- Do not tune the abundance-axis protocol post hoc against SPOP labels unless
  that decision is documented as a new analysis phase.

### Gate 6: DMS Parent-Relative Movement

Question answered:

Do RT-CDS point mutations move a parent construct toward or away from the
selected abundance-like axis, and do large movers localize by protein or
residue features?

Required inputs:

- selected fitted abundance axis;
- DMS rows scored as `sensitivity`;
- complete and unambiguous parent mapping;
- optional mutation annotations, such as residue position, RT domain, mutation
  class, or substitution type.

Outputs and plots:

- `axis_delta` vs `orthogonal_delta`;
- distribution of `axis_delta` by mutation class or domain when annotations
  are reliable;
- ranked positive and negative axis movers;
- optional domain/residue enrichment only after annotation quality is
  established.

Decision rule:

- Axis-moving mutations are candidates for biological interpretation.
- Orthogonal movers may reflect broad representation disruption rather than
  abundance-like shifts.
- Domain or residue localization is interpretive evidence only when annotation
  coverage is reliable.

Failure mode:

- If parent mapping is missing, incomplete, or ambiguous, DMS remains a
  held-out score distribution only.
- Do not present DMS axis movement as a predicted sponging effect without SPOP
  or another functional label.

## Plot and Notebook Surfaces

Use existing plot kinds first:

- `xy_scatter_grid`;
- `distribution_grid`;
- `metric_panel_grid`;
- existing scorecard/frontier surfaces if suitable.

First-pass surfaces:

- trait value vs axis score;
- endpoint sensitivity summary;
- reference/compiler projection ranking;
- Crawford vs Khan axis-score scatter;
- view scorecard;
- DMS axis delta vs orthogonal delta only when parent mapping exists;
- UMAP appendix.

Recommended RT-facing plot or deliverable IDs:

- `rt_lnrna_trait_axis_existence`;
- `rt_lnrna_trait_axis_endpoint_sensitivity`;
- `rt_lnrna_crawford_khan_axis_agreement`;
- `rt_lnrna_trait_axis_view_scorecard`;
- `rt_lnrna_reference_compiler_axis_projection`;
- `rt_lnrna_dms_parent_delta_on_trait_axis`;
- `rt_lnrna_abundance_spop_alignment` when Reader SPOP exists.

The exact IDs may follow existing workspace naming conventions, but each
surface should map cleanly to one gate in the decision funnel. Avoid reusing
the existing candidate frontier as the primary abundance-axis evidence unless
its metrics are rebuilt from trait-axis summaries.

Visible RT labels should use:

- `RT`;
- `CDS`;
- `DMS`;
- `lnRNA`;
- `Eco1`;
- "span" instead of visible "slot";
- "Crawford msDNA abundance" or "Khan RT-DNA abundance" instead of vague
  "prior" when describing source values.

Plot layout must be part of the configured surface, not renderer-specific
special casing. Trait-axis plots should support:

- wrapped or shortened long view labels;
- y-axis label padding for metric panels;
- legend placement outside crowded panels;
- reserved legend margins;
- minimum panel width for multi-view grids;
- reduced title size inside compact scorecards;
- x-tick wrapping or rotation for endpoint-definition labels;
- collision avoidance for value labels and neighboring panels;
- downsampled or aggregated DMS plots by default, with full rows available in
  Parquet.

Notebook order:

1. Dataset denominator.
2. Axis existence.
3. Crawford/Khan concordance.
4. View selection.
5. Reference/compiler projection.
6. DMS sensitivity and parent-relative analysis.
7. Future SPOP readiness or alignment.
8. UMAP appendix.

If the configured review path starts with dataset overview or trait-axis
decision surfaces, the generated notebook should not open by default on a
secondary health panel. Health and UMAP surfaces remain important, but they
should support the decision funnel rather than replace it as the first screen.

### RT Study-Facing Companion Minimum

The RT companion doc should not duplicate this maintainer spec. It should
explain how to read the dogfood outputs in study terms.

Minimum sections:

- biological question and staged decision funnel;
- Crawford source scope and why Crawford abundance is not pooled with Khan;
- Khan source scope and why sparse Khan support is interpreted separately;
- GenBank and compiler MSD as unlabeled reference/candidate overlays;
- DMS as held-out sensitivity rows, with parent-relative interpretation only
  when mapping exists;
- future Reader SPOP alignment and the no-SPOP-claims rule before labels
  materialize;
- visible terminology and label rules for plots;
- current first-pass limitations and deferred extensions.

Do not place generic implementation details, registry internals, or matrix
loading behavior in the RT companion except as needed to explain evidence
limits.

## Performance

The primitive must be batched and chunk-aware.

- Load and normalize each candidate view once per scalar build where practical.
- Fit all configured axes and score all configured populations in one pass per
  view where feasible.
- Use endpoint fitting only; do not run pairwise all-vs-all operations for the
  primary surface.
- Use chunked dot products for large matrices.
- Write row-level projections as Parquet sidecars.
- Keep summary artifacts small.
- Keep notebook controls on manifests, scalar sidecars, plot catalogs, and
  compact summaries. They must not open raw `matrix.npy` payloads.
- Downsample or aggregate heavy DMS plots by default.
- Avoid large Python list-of-dict loops for row-level outputs.
- Do not raise global thread defaults.

Implementation evidence should record scalar runtime, max RSS, raw-matrix read
avoidance for notebook controls, and any slow browser/source-switch profiling.

Performance validation should include:

- baseline RT matrix inventory before implementation: matrix count, total size,
  and largest view;
- runtime and max RSS for the configured RT trait-axis scalar build or a
  representative subset when full RT artifacts are too expensive for local
  iteration;
- confirmation that `workspace snapshot --dry-run` and notebook controls remain
  metadata/freshness-oriented and do not load raw vector matrices;
- browser or notebook interaction profiling if source/view/filter switches
  remain slow after new trait-axis surfaces are added;
- no global BLAS/thread default change unless a separate benchmark justifies
  it.

## Synthetic Tests

Add generic tests independent of RT biology:

- endpoint selection and quantile behavior;
- empty endpoint failure;
- overlapping endpoint failure;
- degenerate centroid failure;
- monotonic synthetic source-value correlation sign;
- endpoint sensitivity rows for min/max, 5 percent, 10 percent, and 20 percent;
- sensitivity rows excluded from fit by default;
- deliberate override behavior if sensitivity rows can fit;
- population-role provenance for fitted trait-axis builders;
- a non-fitted scalar fixture or contract check showing that ordinary
  LatentDNA plots/scalars are not forced to emit trait-axis population
  provenance;
- parent-relative `axis_delta` and `orthogonal_delta`;
- missing or ambiguous parent mapping invalid/skipped status;
- matrix/metadata dimension mismatch failure.

## Validation

Use current command equivalents if CLI names drift:

```bash
uv run latentdna validate workspace --workspace rt_lnrna_sponging_construct_triage --deep --json
uv run latentdna workspace snapshot --workspace rt_lnrna_sponging_construct_triage --json --dry-run
uv run latentdna notebook generate latent_geometry_browser --workspace rt_lnrna_sponging_construct_triage --json
uv run ruff check .
uv run ruff format --check .
uv run python -m dnadesign.devtools.architecture.boundaries --repo-root .
uv run python -m dnadesign.devtools.docs.checks
git diff --check
```

Also run targeted LatentDNA scalar tests, RT study tests touching
metadata/docs/representation, and existing plot/notebook health checks relevant
to regenerated artifacts.

### Reader and Skeptic Checks

When reviewing the implementation, answer these questions from the artifact
itself:

- Can a maintainer identify which rows fit each axis and which rows were only
  scored?
- Can a reviewer tell whether Crawford and Khan were kept source-scoped and not
  numerically pooled?
- Can a reviewer tell whether DMS ever entered fit, and if so, whether that was
  an explicit tested override?
- Can a notebook reader distinguish unsupported, insufficient, source-confounded,
  and view-fragile outcomes?
- Can a future SPOP implementer add a SPOP trait layer by config without
  changing trait-axis projection mechanics?
- Can plot/browser controls work from scalar/catalog sidecars without opening
  raw matrices?

Hostile-but-competent objections the implementation should already address:

- "This is just source-family separation." The summary and plot surfaces should
  make source/provenance diagnostics visible and should not rank views by source
  separation as a positive objective.
- "The axis is a duplicate-density artifact." The implementation should state
  row weighting, exact-duplicate policy, and no near-duplicate collapse.
- "The SPOP conclusion is post hoc." The RT docs should require a frozen or
  versioned abundance-axis protocol before SPOP alignment claims.
- "The notebook proves biology from UMAP." UMAP should be appendix-level and
  never substitute for Gate 1 axis-existence evidence.

## Acceptance Criteria

The first pass is ready for review when:

- the generic runtime contains no RT-specific branch or noun-driven behavior;
- row-level and summary scalar artifacts can be produced from synthetic tests;
- failed endpoint and alignment cases produce explicit errors or invalid
  summaries with reasons;
- the RT workspace can configure Crawford and Khan as independent trait axes;
- GenBank, compiler MSD, and RT-CDS DMS rows are scored only as configured
  overlays;
- DMS rows cannot enter fit by default;
- endpoint sensitivity summaries cover min/max, 5 percent, 10 percent, and
  20 percent definitions;
- plot and notebook surfaces expose the staged decision funnel without making
  UMAP primary evidence;
- generated artifacts, if any, were regenerated through recipes or official
  CLIs;
- validation output records which checks were run and which were deferred.

The first pass is not ready if:

- Crawford and Khan are numerically pooled into one target;
- source-family separation is treated as a positive objective for selecting
  the trait view;
- SPOP claims appear before Reader-owned SPOP labels are materialized;
- DMS perturbation conclusions are made without parent mapping;
- notebook or plot controls read raw matrices interactively;
- generated outputs are hand-edited.

## First-Pass Scope

Include:

- generic row-level trait-axis projection;
- generic summary metrics;
- Crawford axis config;
- Khan axis config;
- endpoint sensitivity summary;
- GenBank/compiler/DMS score populations;
- DMS exclusion from fit by default;
- existing plot-kind surfaces for abundance value vs axis score, endpoint
  sensitivity, reference/compiler projection, Crawford-vs-Khan score scatter,
  and view scorecard;
- generic fitted-geometry provenance.

May defer, but document:

- full null/permutation tests;
- direct axis-vector concordance if integration is not straightforward;
- DMS parent-relative plots if parent mapping is incomplete;
- composition or sequence-similarity baselines;
- domain/residue enrichment for DMS;
- SPOP alignment until Reader-owned labels exist.

## Future Extensions

These are useful but should not block the first pass.

### Null Controls

Potential generic nulls:

- shuffled source values within trait/source scope;
- random endpoint groups with the same endpoint sizes;
- source-family or provenance axes as nuisance comparisons;
- DMS-excluded versus DMS-included sensitivity checks;
- matched sequence-similarity controls where stable metadata exists.

Null controls should land only if they can be implemented generically and
efficiently. Avoid adding RT-only null logic to the primitive.

### Composition and Sequence-Similarity Overlays

Potential overlays:

- GC content;
- k-mer composition;
- MSD length;
- stem-loop length;
- compiler primitive labels;
- edit distance to selected references;
- predicted structure;
- RT domain, residue, or mutation class annotations.

Use these as confound/reference overlays when stable upstream fields exist. Do
not add unstable ad hoc extraction in the first trait-axis implementation.

### Reader SPOP Trait Layer

When Reader SPOP labels exist:

- configure SPOP as a separate trait layer;
- preserve label coverage provenance;
- compare SPOP metric to selected abundance-axis score;
- optionally fit a SPOP trait axis with the same primitive;
- report abundance/SPOP alignment outcomes explicitly.

Do not invent SPOP labels, do not treat abundance as a SPOP proxy without
evidence, and do not use SPOP to retroactively tune the abundance-axis protocol
without documenting that decision.

## Claim Ladder

The RT study or notebook should be able to classify each trait/view result as:

- `supported`: abundance value correlates with axis score and endpoint
  separation/stability checks are acceptable.
- `source_confounded`: signal appears dominated by provenance/source family.
- `view_fragile`: signal appears only in one unstable or implausible view.
- `not_supported`: no monotonic or endpoint separation.
- `insufficient`: too few rows, degenerate endpoints, missing source values, or
  invalid alignment.

These labels do not need to be hard-coded in the first pass, but the summary
metrics must be sufficient to support them.

## Open Questions

- Should endpoint sensitivity defaults be generic LatentDNA defaults or
  workspace config defaults?
- Should direct axis-vector concordance land in the first implementation slice?
- Is DMS parent mapping complete enough for parent-relative plots in the first
  RT dogfood pass?
- Which composition or sequence-similarity fields already exist as stable
  metadata, if any?
- Should first pass be row-weighted or subject-weighted? Default to row-weighted
  unless existing contracts define subject weighting.
- Does the source sample size require fit rows to double as fit-eval summaries?
  If yes, provenance must say so.

## Implementation Surfaces

- Generic builders:
  `src/dnadesign/latentdna/src/scalars/builders/trait_axis_projection.py`
- Scalar dispatch:
  `src/dnadesign/latentdna/src/scalars/build.py`
- Generic contract tests:
  `src/dnadesign/latentdna/tests/test_trait_axis_projection.py`
- RT configuration and plot semantics:
  `src/dnadesign/latentdna/workspaces/rt_lnrna_sponging_construct_triage/`
- RT study interpretation:
  `docs/studies/rt_lnrna_sponging_construct_triage/contexts/latentdna/trait-axis-projection.md`

Keep runtime, study configuration, and generated artifacts reviewable as
separate changes. Regenerate scalar, plot, notebook, and deliverable artifacts
only through workspace recipes or official CLIs; do not hand-edit outputs.

Preserve the division:

```text
LatentDNA fits and audits generic geometry.
RT-lnRNA interprets biological cohorts and labels.
OPAL waits for selected X and real labels.
```
