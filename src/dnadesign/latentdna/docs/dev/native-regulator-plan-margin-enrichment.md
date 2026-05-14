# Native Regulator Plan-Margin Enrichment

**Owner:** dnadesign-maintainers
**Status:** table-builder implemented; static plot planned
**Last verified:** 2026-05-14
**Study:** `stress_ethanol_cipro_growth`

This spec defines a LatentDNA appendix deliverable for exploratory RegulonDB
regulator enrichment among native promoter-core60 landmarks after those cores
are embedded in the same pDual10 1 kb context surface as the synthetic promoter
study.

The deliverable is intentionally named around **plan margins**, not only tails.
The tail views are the readable summary, but the durable quantity is the
margin between a native row and the synthetic-plan centroids.

Current implementation posture:

- The first code slice emits the contract tables as one scalar artifact:
  `table.parquet` for regulator enrichment plus
  `native_plan_margin_scores.parquet` and
  `native_plan_margin_tail_membership.parquet` side tables.
- Static plot promotion and notebook panel wiring should consume those tables
  in a later renderer/notebook slice. The notebook must not recompute the
  statistics inline.

## Purpose

Ask this narrow question:

> Among parent-resolved native RegulonDB core60 promoters embedded in pDual10
> context, which curated regulator associations are overrepresented in the
> native rows most specific to each synthetic plan centroid?

This is an exploratory interpretation layer for the existing LatentDNA
representation-triage surface. It is not an OPAL input and must not be used as
a candidate-selection rule.

The claim boundary is:

> RegulonDB association enrichments among native promoter-core60 landmarks in
> synthetic-plan latent margins.

Do not claim:

> Mechanistic labels for synthetic promoters, validated ethanol biology, or
> complete native regulatory logic preserved in the pDual10 transfer.

## Naming Semantics

Use stable artifact names that describe the statistical operation rather than
the historical motivation.

| Surface | Artifact id | Reviewer-facing label | Semantics |
| --- | --- | --- | --- |
| Existing pre-specified landmark audit | `native_tf_axis_orientation_audit` for backward compatibility; future alias may be `native_stress_regulator_landmark_audit` | BaeR/CpxR/LexA regulator landmark audit | Tests only the pre-specified BaeR/CpxR ethanol-side and LexA SOS-side landmark bins. |
| New exploratory appendix | `native_regulator_plan_margin_enrichment` | Native regulator enrichment in synthetic-plan margins | Tests all source-backed RegulonDB regulators for enrichment in plan-specific native margins and tails. |

Do not collapse these surfaces. The first asks whether pre-specified landmark
regulators land in expected directions. The second asks which regulator
associations appear after selecting native rows by synthetic-plan geometry.

If the existing landmark artifact is renamed later, do it as a compatibility
migration with an alias and route/status update. Do not rename generated
artifacts in the same slice as the new enrichment feature.

## Non-Goals

- No OPAL, SFXI, or active-learning acquisition coupling.
- No candidate filtering or candidate ranking.
- No LLM-generated ontology or free-text regulator interpretation.
- No broad RegulonDB EDA, sigma-factor atlas, GO expansion, or iModulon layer.
- No mutation of Infer sidecars, DenseGen source rows, or existing 157k
  synthetic vector artifacts.
- No hand-edits under generated `outputs/**`.

## Input Contracts

The implementation should fail fast when any required input is missing,
ambiguous, or inconsistent.

Required row universe:

- Native rows come from the current stress-study merged quota, filtered to
  `derived__parent_dataset == "usr_regulondb_native_promoters"`.
- Current expected denominator is `3180` parent-resolved rows, matching the
  current BaeR/CpxR/LexA landmark audit contract. Treat this as a
  workspace-configured expectation, not a package constant.
- Each row must have exactly one native parent id. Missing or duplicate parent
  lineage is a hard error unless a future config explicitly declares an
  exclusion list and records the excluded ids.

Required embedding view:

- Primary view:
  `intermediate_embedding_7b_context_anchor_mean_bidir_concat`.
- Rows are the RegulonDB core60 anchors inserted into the pDual10 1 kb context
  and represented by the same forward/RC anchor-mean bidirectional concat used
  for candidate-X triage.
- The view must use the same block-standardized, row-L2-normalized geometry
  contract as the rest of the study-facing centroid/margin surfaces.

Required centroid groups:

- `background`: DenseGen `background_only`.
- `ethanol`: DenseGen `ethanol`.
- `cipro`: DenseGen `ciprofloxacin`.
- `dual`: DenseGen `ethanol_ciprofloxacin`.

Required regulator source:

- `usr_regulondb_native_promoters/_relations/regulatory_interactions.parquet`.
- Required columns include native parent id, regulator abbreviation, source
  route or release, and enough evidence fields to audit provenance.
- Regulator membership is promoter-level binary membership. A promoter with
  multiple interaction rows for one regulator counts once for that regulator.

Required validation:

- The native row set, embedding row set, and regulatory sidecar must agree by
  parent id.
- Missing required columns are hard errors.
- Zero matched regulators is a hard error.
- Unsupported threshold, FDR, or permutation configuration is a hard error.
- Silent fallback to the old BaeR/CpxR/LexA-only audit is forbidden.

## Artifact Ontology

The deliverable should emit small, inspectable appendix tables with explicit
semantics.

### `native_plan_margin_scores`

One row per parent-resolved native promoter.

Minimum columns:

- `native_parent_id`
- any configured `native_metadata_columns` copied from the view row table
- `embedding_view`
- `sim_<plan_id>` for each configured plan id
- `margin_<plan_id>` for each configured plan id
- `nearest_plan`
- `nearest_plan_tie_count`
- `regulator_degree`

### `native_plan_margin_tail_membership`

One row per native promoter, plan, threshold, and tail mode when the row is in
that tail.

Minimum columns:

- `native_parent_id`
- `plan`
- `threshold`
- `tail_mode`
- `rank`
- `margin`
- `nearest_plan`

Primary `tail_mode` is `margin_top_quantile`.
Optional sensitivity `tail_mode` is `margin_top_quantile_nearest_plan_only`.

### `native_regulator_plan_margin_enrichment`

One row per regulator, plan, threshold, and tail mode test.

Minimum columns:

- `regulator_abbrev`
- `plan`
- `threshold`
- `tail_mode`
- `n_total_native`
- `n_tail`
- `n_regulator_total`
- `n_regulator_tail`
- `tail_fraction`
- `background_fraction`
- `enrichment_ratio`
- `odds_ratio`
- `p_value`
- `q_value`
- `p_value_method`
- `fdr_method`
- `passes_min_support`
- `passes_min_tail_hits`
- `is_common_regulator`
- `notes`

Future-only artifact:

- `native_regulator_function_enrichment`, only after source-backed functional
  tags or an external ontology mapping are introduced by contract.

## Math Contract

Let `x_i` be the normalized vector for native promoter row `i`, and let
`mu_p` be the normalized centroid for synthetic plan `p`.

Plans are configured by `centroid_groups`. The stress study currently declares
`background`, `ethanol`, `cipro`, and `dual`, but LatentDNA must treat plan ids
as artifact config, not as hard-coded primitive semantics. Plan ids must be
stable column-safe identifiers because the score table emits `sim_<plan_id>` and
`margin_<plan_id>` columns.

```text
P = ordered keys from centroid_groups or explicit plan_order
```

Cosine similarity:

```text
sim(i, p) = cosine(x_i, mu_p)
```

Plan margin:

```text
margin(i, p) = sim(i, p) - max_{q in P, q != p} sim(i, q)
```

Nearest plan:

```text
nearest_plan(i) = argmax_{p in P} sim(i, p)
```

Ties use the declared plan order for deterministic assignment, and the score
table reports `nearest_plan_tie_count` so tied rows remain auditable.

Primary tails:

- top `5%` by `margin(i, p)` for each plan
- top `10%` by `margin(i, p)` for each plan

Optional sensitivity tails:

- same rank thresholds, additionally requiring `nearest_plan(i) == p`

Use rank or quantile tails rather than standard-deviation cutoffs because the
margin distributions are not assumed normal, and standard-deviation thresholds
invite post-hoc tuning.

## Enrichment Contract

For each regulator and each plan-tail set, build a promoter-level 2x2 table:

| | In tail | Not in tail |
| --- | ---: | ---: |
| Has regulator | `a` | `b` |
| Does not have regulator | `c` | `d` |

Report:

- raw count: `a`
- tail fraction: `a / (a + c)`
- background fraction: `(a + b) / (a + b + c + d)`
- enrichment ratio: `tail_fraction / background_fraction`
- odds ratio
- p-value from Fisher exact or a declared hypergeometric equivalent
- Benjamini-Hochberg FDR q-value within the configured test family

Default primary filters:

- `min_global_promoters = 10`
- `min_tail_hits = 3`
- thresholds `[0.05, 0.10]`
- FDR method `benjamini_hochberg`

Rare regulators can remain in the output table with
`passes_min_support == false`, but they should not be promoted as primary
interpretation claims.

Common global regulators such as `CRP`, `FNR`, `Fis`, `H-NS`, `IHF`, `Fur`,
`Lrp`, and `ArcA` should remain visible, but they need a separate
`is_common_regulator` interpretation flag. Their raw counts are usually not
interesting unless the enrichment ratio and FDR-adjusted test show enrichment
above background prevalence.

Annotation-density sensitivity:

- Compute each promoter's `regulator_degree` as the number of unique regulators
  associated with that native parent.
- Add a secondary permutation or matched-null sensitivity that stratifies or
  matches by `regulator_degree`.
- This sensitivity is not required for the first table to exist, but the
  configuration must expose whether it was run and the output must not imply it
  ran when it did not.

## Plot Contract

The first plot should be compact and appendix-scoped.

Recommended surface:

- x-axis: `margin_ethanol` or `sim_ethanol - max(other plans)` when plan is
  fixed; alternatively use a small-multiple plan-margin panel.
- y-axis: `margin_cipro` or plan-specific margin depending on the selected
  panel.
- background: all parent-resolved native RegulonDB core60-in-pDual10 rows as
  faint points or density.
- overlay: tail membership at 5% and 10%.
- labels: top enriched regulators per plan and threshold after min-support and
  FDR filters.
- optional companion table: regulator counts and enrichment statistics.

Do not make the enrichment plot look like a candidate-selection chart. It is an
appendix interpretation surface.

## Notebook Accordion Requirements

The generated Marimo notebook panel should include an `mo.accordion` section
with these reader-facing subsections.

1. **What This Tests**

   This plot projects native RegulonDB promoter-core60 landmarks into the same
   pDual10 context-anchor latent geometry used for synthetic promoter
   triage. It asks which curated regulator associations are enriched among the
   native rows most specific to each synthetic plan centroid.

2. **Margin Math**

   Show the equations for `sim(i, p)`, `margin(i, p)`, and `nearest_plan(i)`.
   State that centroids are derived from synthetic DenseGen plan groups and
   applied to native pDual10-embedded core60 rows.

3. **Why Rank Tails**

   Explain that top 5% and 10% tails are rank-based because margin
   distributions are not assumed normal. These thresholds are fixed by config
   and are not tuned to produce a preferred biology result.

4. **How Regulators Are Counted**

   State that each native promoter counts at most once per regulator, even if
   RegulonDB contains multiple interaction rows for that regulator-promoter
   pair.

5. **How Common Regulators Are Controlled**

   Explain that the table reports background prevalence, enrichment ratio,
   odds ratio, p-value, and FDR q-value. Common regulators are not suppressed,
   but raw abundance alone is not treated as evidence.

6. **Caveats**

   State that the rows are native core60 windows transferred into pDual10
   context. Regulatory sites outside core60, genomic context, condition
   specificity, and indirect regulatory logic may be missing. Therefore the
   result is exploratory association enrichment, not a mechanistic annotation
   of synthetic promoters.

7. **Study Boundary**

   State that this appendix does not feed OPAL and does not change the
   candidate-X decision contract.

## Module Placement

Keep implementation modular and avoid growth in `scalars/build.py`.

Recommended code organization:

```text
src/dnadesign/latentdna/src/enrichments/
  table_contracts.py                   # generic table/config validation helpers
  enrichment_stats.py                  # generic hypergeometric, odds-ratio, and FDR helpers
  categorical_enrichment.py            # generic categorical-feature enrichment primitive
  regulatory_plan_margin.py            # artifact-specific plan/tail adapter
  regulatory_plan_margin_contracts.py  # artifact config/schema helpers and output dataclasses

src/dnadesign/latentdna/src/scalars/builders/
  regulatory_plan_margin.py            # scalar I/O orchestration for the artifact

src/dnadesign/latentdna/src/plots/
  regulatory_plan_margin_enrichment.py  # plot table transforms/rendering helpers
```

`scalars/build.py` may own a thin dispatch entry only. It must not parse this
artifact's full config or write its side tables directly.

Study-specific vocabulary belongs in the adapter/config layer. Generic
LatentDNA primitives must not depend on RegulonDB, regulator, BaeR/CpxR/LexA,
or `stress_ethanol_cipro_growth` semantics. In particular, categorical
enrichment should be reusable for any subject universe, feature membership
sidecar, and named row groups.

The scalar builder module owns:

- workspace-local path resolution
- table/matrix loading
- config-shape validation before calling the artifact builder
- writing declared scalar side tables
- returning `BuiltScalarArtifact` metadata

Notebook code must only render configured artifacts and explanatory text. It
must not compute enrichment statistics inline.

## Workspace Config Shape

Add a first-class appendix deliverable after the existing landmark audit.

Example config shape:

```yaml
native_regulator_plan_margin_enrichment_recipe:
  kind: native_regulator_plan_margin_enrichment
  embedding_view: intermediate_embedding_7b_context_anchor_mean_bidir_concat
  native_filter:
    column: derived__parent_dataset
    equals: usr_regulondb_native_promoters
  expected_output_rows: 3180
  native_metadata_columns:
    - alias_id
    - regulondb__primary_promoter_id
    - regulondb__primary_promoter_name
  centroid_groups:
    background: [background_only]
    ethanol: [ethanol]
    cipro: [ciprofloxacin]
    dual: [ethanol_ciprofloxacin]
  # Optional; defaults to centroid_groups insertion order.
  plan_order: [background, ethanol, cipro, dual]
  regulatory_interactions:
    dataset: usr_regulondb_native_promoters
    path: _relations/regulatory_interactions.parquet
    join_key: derived__parent_id
    regulator_column: regulator_abbrev
    required_columns:
      - source_release
      - source_route
      - regulatory_interaction_id
      - confidence
      - evidence
  thresholds: [0.05, 0.10]
  min_global_promoters: 10
  min_tail_hits: 3
  fdr_method: benjamini_hochberg
  tail_modes:
    - margin_top_quantile
    - margin_top_quantile_nearest_plan_only
  common_regulators:
    - CRP
    - FNR
    - Fis
    - H-NS
    - IHF
    - Fur
    - Lrp
    - ArcA
```

## Acceptance Checks

Unit tests:

- plan similarity and margin math
- nearest-plan tie behavior is declared and deterministic
- top-quantile tail membership
- nearest-plan-only sensitivity tail membership
- promoter-level regulator deduplication
- Fisher or hypergeometric p-value calculation
- Benjamini-Hochberg q-value calculation
- min-support and min-tail-hit flags
- zero-regulator and rare-regulator behavior

Contract tests:

- missing regulatory sidecar fails
- missing required sidecar columns fails
- missing embedding view fails
- unexpected native row count fails when configured
- duplicate parent ids fail unless an explicit exclusion list is configured
- unsupported threshold config fails
- unsupported FDR method fails
- output rows carry method metadata

Workspace checks:

```sh
uv run pytest -q src/dnadesign/latentdna/tests/test_categorical_enrichment.py
uv run pytest -q src/dnadesign/latentdna/tests/test_regulatory_plan_margin_enrichment.py
uv run pytest -q src/dnadesign/latentdna/tests/test_scalar_build.py -k native_regulator_plan_margin_enrichment
uv run latentdna validate workspace --workspace stress_ethanol_cipro_growth --deep
uv run latentdna generate notebook --workspace stress_ethanol_cipro_growth
uv run python -m dnadesign.devtools.docs.checks
```

Generated artifacts should be produced by workspace commands only. Do not
commit generated `outputs/**` unless the PR scope explicitly includes artifact
refresh and the large-file review is intentional.

## Open Questions

- Whether `min_global_promoters` should default to `10` or `20` for the first
  plotted label set. The table can compute both, but the displayed top labels
  should use one predeclared threshold.
- Whether degree-stratified permutation should be required before showing
  q-value-ranked labels in the notebook, or allowed as a second-pass
  sensitivity column.
- Whether the existing artifact id `native_tf_axis_orientation_audit` should be
  migrated to `native_stress_regulator_landmark_audit` after the new appendix
  lands, or left stable with only the reviewer-facing label changed.
