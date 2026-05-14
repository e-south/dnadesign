# Native Regulator Plan-Margin Enrichment

This appendix deliverable is an exploratory interpretation layer for the
stress promoter study. It asks which source-backed RegulonDB regulator
associations are enriched among native promoter-core60 rows that sit in high
synthetic-plan latent margins after transfer into the study pDual10 1 kb
context.

It is not an OPAL input, not a candidate-selection rule, and not a mechanistic
annotation of synthetic promoters.

## Contract

- Geometry: `intermediate_embedding_7b_context_anchor_mean_bidir_concat`
- Native rows: `derived__parent_dataset == usr_regulondb_native_promoters`
- Expected native denominator: `3180` parent-resolved rows
- Regulator source:
  `usr_regulondb_native_promoters/_relations/regulatory_interactions.parquet`
- Centroid groups: background, ethanol, ciprofloxacin, and dual-stress DenseGen
  plans
- Primary thresholds: top 5% and top 10% by plan margin
- Regulator counting: promoter-level binary membership, not interaction-row
  counts

The margin for plan \(p\) is:

$$
\mathrm{margin}(i,p)=\cos(x_i,\mu_p)-\max_{q \neq p}\cos(x_i,\mu_q).
$$

## Outputs

The first implementation emits one scalar artifact directory:

- `table.parquet`: regulator-by-plan enrichment tests
- `native_plan_margin_scores.parquet`: one row per native promoter with
  similarity, margin, nearest-plan, and regulator-degree columns
- `native_plan_margin_tail_membership.parquet`: promoter-plan-threshold tail
  membership rows

The enrichment table reports raw counts, tail fraction, background fraction,
enrichment ratio, odds ratio, one-sided hypergeometric survival p-value,
Benjamini-Hochberg q-value, and min-support flags.

The companion plot is a static appendix summary over the configured top 10%
margin tail. It shows support-filtered top regulator associations per plan as
enrichment-ratio bars, with q-values annotated where available and common
global regulators visually separated from other regulators.

### native_regulator_plan_margin_enrichment | Native regulator plan-margin enrichment

#### Plot details

**Data.** Parent-resolved RegulonDB native core60 rows embedded in the study
pDual10 context-anchor view, joined to promoter-level regulator memberships
from `usr_regulondb_native_promoters/_relations/regulatory_interactions.parquet`.

**Preprocessing.** Synthetic plan centroids are computed from DenseGen plan
groups. Native rows are scored by plan margin, assigned to fixed 5% and 10%
rank tails, and regulator enrichment is counted once per native promoter per
regulator.

**Definition.** The plotted bar length is enrichment ratio: tail regulator
prevalence divided by native-background regulator prevalence. The plot uses the
configured 10% `margin_top_quantile` tail and displays rows passing the
minimum-support and minimum-tail-hit filters.

**Decision use.** This is an appendix interpretation surface. It can suggest
which curated native regulator associations are overrepresented in synthetic
plan-margin tails, but it is not an OPAL input and not a candidate-selection
rule.

**Limits.** The plot does not claim that a transferred core60 sequence retains
complete native regulatory logic. q-values, support counts, and the common
regulator flag must be read with the table before making any biology-facing
claim.

## Interpretation

Use this as a hypothesis-generating biology-facing appendix. A result such as
"CpxR-associated promoters are enriched in the cipro-margin tail" means only
that curated native promoter associations are overrepresented in a latent tail
under the pDual10 core60 transfer. It does not prove the transferred 60 bp
sequence retains complete native regulatory logic or that the synthetic
promoters have that mechanism.

Common regulators remain visible but should be interpreted through enrichment
over background prevalence, not raw counts alone.

## Current State

The scalar table-builder, static plot renderer, workspace recipe, and generated
notebook plot-review panel are configured. Marimo consumes the persisted table,
plot artifact, and plot-semantics sidecar; it does not recompute enrichment
statistics inline.
