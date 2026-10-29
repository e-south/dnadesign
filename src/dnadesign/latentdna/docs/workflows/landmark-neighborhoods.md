<!--
**Owner:** dnadesign-maintainers
**Last verified:** 2026-10-28
-->

[LatentDNA guide](../README.md) / [Workflows](README.md)

# Summarize landmark neighborhoods

Use neighborhood enrichment to examine the metadata composition around declared reference sequences. It combines a saved nearest-neighbor graph with a cohort column, such as an experimental group or sequence class.

## Required inputs

- A materialized view and a neighbor artifact fitted on the intended row population.
- Explicit landmark membership in that population.
- A `kind: column` cohort declaration pointing to the metadata to summarize.

Fit neighbors with `neighbors fit`, then pass that artifact, the cohort and the reference to `enrich score`. For example, with existing `embedding_knn`, `groups` and `reference` declarations:

```bash
# Summarize cohort membership around the selected reference rows.
latentdna enrich score reference_neighborhoods \
  --workspace /path/to/workspace --neighbors embedding_knn \
  --cohort groups --landmark reference
```

The output table under `outputs/enrichments/reference_neighborhoods/` records each cohort’s fraction among neighbor hits and in the full neighbor population. Their ratio is `enrichment_ratio`; their difference is `enrichment_delta`. A ratio above one indicates that the cohort occurs more often among those hits than in the comparison population. A row that neighbors several reference members contributes once for each hit.

Render a heatmap from the saved `enrichment_set`, or compare neighborhood structure across views using [cross-view agreement](cross-view-agreement.md). Keep the population, neighbor count and metric explicit when interpreting enrichment. See [landmark concepts](../concepts/landmarks.md) for reference definitions.
