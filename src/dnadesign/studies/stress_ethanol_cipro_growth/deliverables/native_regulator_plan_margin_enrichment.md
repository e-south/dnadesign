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

The table-builder and workspace recipe are configured. Static plot promotion
and notebook-panel wiring should be implemented as a later renderer/notebook
slice that consumes the emitted tables instead of recomputing statistics in
Marimo.
