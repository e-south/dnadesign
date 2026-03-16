## cluster docs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-16

This docs surface covers exploratory clustering and UMAP work after one explicit feature column is already available.

### Start here

1. If you do not yet have one explicit `infer__...` column chosen as `X`, return to [promoter characterization feature matrix](../../usr/docs/operations/promoter-characterization-feature-matrix.md).
2. If you already have a chosen `X` column and want the first runnable path, open [exploratory clustering workflow](workflows/exploratory-clustering.md).
3. If you need flag, preset, job, OPAL-join, or results semantics before running, open [cluster CLI contracts](reference/cli-contracts.md).
4. If you need the downstream ownership split versus USR, infer, and OPAL, open [cluster ownership boundary](concepts/ownership-boundary.md).

### Documentation by workflow

#### Run exploratory clustering and UMAPs
- [exploratory clustering workflow](workflows/exploratory-clustering.md): first `fit -> umap -> analyze` path over one chosen `X` column.

#### Reuse OPAL outputs in exploratory plots
- [exploratory clustering workflow](workflows/exploratory-clustering.md#optional-opal-join-path): attach OPAL objective or prediction columns for coloring and summaries.
- [cluster CLI contracts](reference/cli-contracts.md#opal-join-contract): exact `--opal-*` requirements.

#### Maintain jobs, presets, and results
- [cluster CLI contracts](reference/cli-contracts.md#jobs-presets-and-results-layout): job/preset layout and precedence rules.
- [cluster CLI contracts](reference/cli-contracts.md#results-and-artifacts): run store, reuse, and cleanup behavior.

#### Return to adjacent tool-owned routes
- [promoter characterization feature matrix](../../usr/docs/operations/promoter-characterization-feature-matrix.md): upstream durable data-plane handoff that creates the chosen `X` column.
- [USR dataset with infer-derived X -> OPAL active learning](../../opal/docs/workflows/usr-infer-x-active-learning.md): supervised downstream branch once exploratory clustering is no longer the next task.

### Documentation by type

- [docs index by type](index.md)
- [workflow](workflows/exploratory-clustering.md)
- [reference](reference/cli-contracts.md)
- [concept](concepts/ownership-boundary.md)
- [package entrypoint](../README.md)
- [repository docs index](../../../../docs/README.md)
