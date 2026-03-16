## cluster docs index

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-16

Use this page when you want the cluster docs split by document type rather than by task.

### Read order

1. [Top README](../README.md): package boundary and fast route map.
2. [Docs index by workflow](README.md): task-first router once a chosen feature definition already exists.
3. [Workflow](workflows/exploratory-clustering.md): runnable clustering and UMAP sequence.
4. [Reference](reference/cli-contracts.md): command, layout, OPAL-join, and troubleshooting contracts.
5. [Verification reference](reference/verification.md): deterministic package-local preflight/run/verify path.
6. [Concept](concepts/ownership-boundary.md): ownership split versus USR, infer, and OPAL.
7. [Semantic concept](concepts/semantic-surface.md): package nouns and runtime boundary rules.

### Documentation by type

- [workflow](workflows/exploratory-clustering.md): first runnable exploratory path.
- [reference](reference/cli-contracts.md): flag and data-shape contracts.
- [verification reference](reference/verification.md): maintainer verification loop for contract or docs changes.
- [concept](concepts/ownership-boundary.md): ownership and handoff semantics.
- [semantic concept](concepts/semantic-surface.md): package ontology and artifact roles.
- [repository docs index](../../../../docs/README.md): cross-tool routes back into USR, infer, and OPAL.

### Cross-tool handoffs

- [promoter characterization feature matrix](../../usr/docs/operations/promoter-characterization-feature-matrix.md): upstream source-of-truth and infer write-back path.
- [infer docs](../../infer/docs/README.md): feature-generation router when the chosen feature definition does not exist yet.
- [USR dataset with infer-derived X -> OPAL active learning](../../opal/docs/workflows/usr-infer-x-active-learning.md): switch here when the next step is supervised label/train/select rather than exploratory clustering.
