<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[LatentDNA guide](../README.md) / [Workflows](README.md)

# Compare sequence contexts

Compare two representations of the same sequence or focal region to measure how its numerical representation changes with context. For example, one view may represent an insert alone and another the insert within surrounding DNA.

## Prepare paired inputs

Declare both views and an alignment on their shared subject key. Direct cosine, L2 or vector-difference comparisons require matching dimensions and a compatible `coordinate_space_id`. Matching sequence IDs alone does not make different model spaces comparable.

1. Materialize both views.
2. Build the declared `alignment_set` to record paired rows and any aggregation.
3. Compute the configured paired metrics, such as `context_self_cosine` and `context_shift_l2`, through the workspace recipe.
4. Persist the matched sample IDs and render summaries from those saved artifacts.

The output describes representation change for the paired population. Its interpretation depends on the input sequence and pooling span: a whole-context mean and an insert-only mean summarize different model responses. Neither by itself identifies a regulatory mechanism.

For views in incompatible coordinate spaces, use [cross-view agreement](cross-view-agreement.md) instead. See the [view](../reference/view-contract.md), [alignment](../reference/alignment-contract.md) and [workspace](../reference/workspace-schema.md) references for configuration.
