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

The scalar artifact directory contains:

- `table.parquet`: regulator-by-plan enrichment tests
- `native_plan_margin_scores.parquet`: one row per native promoter with
  similarity, margin, nearest-plan, and regulator-degree columns
- `native_plan_margin_tail_membership.parquet`: promoter-plan-threshold tail
  membership rows
- `native_regulator_plan_rank_tests.parquet`: regulator-by-plan rank tests over
  the full native denominator

The enrichment table reports raw counts, tail fraction, background fraction,
enrichment ratio, odds ratio, one-sided hypergeometric survival p-value,
Benjamini-Hochberg q-value, and min-support flags.

The companion BioCyc biological-process scalar emits:

- `table.parquet`: GO-term-by-plan tail enrichment tests
- `plan_margin_feature_rank_tests.parquet`: GO-term-by-plan rank tests over the
  full native denominator

The companion plot is a static appendix summary over the configured top 10%
margin tail. It shows support-filtered top regulator associations per plan as
enrichment-ratio bars, with q-values annotated where available and common
global regulators visually separated from other regulators. The rank-test table
is the more stable statistical backbone; the plotted tail summary is the
readable appendix view.

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

### native_regulator_go_bp_plan_margin_enrichment | Native regulator GO biological-process margin enrichment

#### Plot details

**Data.** The same native plan-margin score and tail-membership tables, joined
to BioCyc KB 29.6 GO biological-process terms carried by associated RegulonDB
regulators.

**Preprocessing.** A native promoter counts once for a GO term when at least
one associated regulator carries that source-backed term. Obsolete GO terms are
excluded by config.

**Definition.** The plotted bar length is enrichment ratio: tail term
prevalence divided by native-background term prevalence. The plot uses the
configured 10% `margin_top_quantile` tail and displays rows passing the
minimum-support and minimum-tail-hit filters.

**Decision use.** This is a functional-label companion to the regulator plot.
It summarizes source-backed regulator annotations in latent tails; it is not an
OPAL input and not a mechanism claim for synthetic promoters.

**Limits.** GO terms annotate native regulators or regulator genes, not the
transferred core60 sequence itself. Broad transcription terms and common
regulators can dominate, so rank-test and tail tables should be read together.

## Interpretation

Use this as a hypothesis-generating biology-facing appendix. A result such as
"CpxR-associated promoters are enriched in the ciprofloxacin-margin tail" means
only that curated native promoter associations are overrepresented in a latent
tail under the pDual10 core60 transfer. It does not prove the transferred 60 bp
sequence retains complete native regulatory logic or that the synthetic
promoters have that mechanism.

Common regulators remain visible but should be interpreted through enrichment
over background prevalence, not raw counts alone.

## Current State

The scalar table-builder, static plot renderer, workspace recipe, and generated
notebook plot-review panel are configured. Marimo consumes the persisted table,
plot artifact, and plot-semantics sidecar; it does not recompute enrichment
statistics inline.
