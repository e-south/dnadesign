## cluster ownership boundary

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-16

Use this page when you need the package-role decision before choosing a downstream route.

### What cluster owns

- unsupervised clustering over one chosen feature matrix
- UMAP visualization and hue-based exploratory plotting
- unsupervised summaries such as composition, diversity, differential, and numeric analysis
- optional joins of OPAL outputs for exploratory coloring or summarization

### What cluster does not own

- upstream USR source-of-truth assembly
- construct-based context expansion
- infer feature generation or the choice of which upstream representation should become `X`
- supervised label/train/select loops

### Preconditions

- one explicit `infer__...` column is already present and chosen as `X`
- the dataset or file already satisfies any sequence/hue prerequisites for the views you want

### Choose cluster when

- you want exploratory structure before modeling
- you need Leiden clusters, UMAPs, or cluster-level summaries
- you want to color exploratory views with OPAL outputs without entering a supervised loop

### Do not stay in cluster when

- the chosen `X` column does not exist yet: return to [promoter characterization feature matrix](../../../usr/docs/operations/promoter-characterization-feature-matrix.md) or [infer docs](../../../infer/docs/README.md)
- the next task is supervised label/train/select: continue with [USR dataset with infer-derived X -> OPAL active learning](../../../opal/docs/workflows/usr-infer-x-active-learning.md)

### Related docs

- [cluster docs by workflow](../README.md)
- [exploratory clustering workflow](../workflows/exploratory-clustering.md)
- [cluster CLI contracts](../reference/cli-contracts.md)
