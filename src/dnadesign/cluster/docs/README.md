## cluster docs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-16

This is the comprehensive index for `cluster`.
It covers exploratory clustering and UMAP work after one explicit feature definition is already available, plus maintainer verification and public API routes.

### Start here

1. If you do not yet have one explicit chosen feature definition, return to [promoter characterization feature matrix](../../usr/docs/operations/promoter-characterization-feature-matrix.md) for the repository's canonical infer-backed route.
2. If you already have a chosen feature definition and want the first runnable path, open [exploratory clustering workflow](workflows/exploratory-clustering.md).
3. If you need flag, workspace, preset, OPAL-join, or results semantics before running, open [cluster CLI contracts](reference/cli-contracts.md).
4. If you need the public in-process automation boundary, open [`dnadesign.cluster.api`](../api.py).
5. If you need the downstream ownership split versus USR, infer, and OPAL, open [cluster ownership boundary](concepts/ownership-boundary.md) and [cluster semantic surface](concepts/semantic-surface.md).
6. If you are changing `cluster` itself, open [cluster verification contract](reference/verification.md).

### Documentation by workflow

#### Run exploratory clustering and UMAPs
- [exploratory clustering workflow](workflows/exploratory-clustering.md): first `fit -> umap -> analyze` path over one chosen feature definition.

#### Reuse OPAL outputs in exploratory plots
- [exploratory clustering workflow](workflows/exploratory-clustering.md#optional-opal-join-path): attach OPAL objective or prediction columns for coloring and summaries.
- [cluster CLI contracts](reference/cli-contracts.md#opal-join-contract): exact `--opal-*` requirements.

#### Maintain workspaces, presets, and results
- [cluster CLI contracts](reference/cli-contracts.md#workspaces-presets-and-results-layout): workspace/preset layout and precedence rules.
- [cluster CLI contracts](reference/cli-contracts.md#workspace-utilities): workspace discovery and scaffolding commands.
- [cluster CLI contracts](reference/cli-contracts.md#results-and-artifacts): immutable run layout, reuse, and cleanup behavior.
- [`dnadesign.cluster.api`](../api.py): public ad hoc and workspace execution helpers for package-to-package automation.
- [cluster verification contract](reference/verification.md): deterministic preflight/run/verify loop for package changes.

#### Return to adjacent tool-owned routes
- [promoter characterization feature matrix](../../usr/docs/operations/promoter-characterization-feature-matrix.md): upstream durable data-plane handoff that creates one chosen feature definition.
- [USR dataset with infer-derived X -> OPAL active learning](../../opal/docs/workflows/usr-infer-x-active-learning.md): supervised downstream branch once exploratory clustering is no longer the next task.

### Documentation by type

- [docs index by type](index.md)
- [workflow](workflows/exploratory-clustering.md)
- [reference](reference/cli-contracts.md)
- [verification reference](reference/verification.md)
- [concept](concepts/ownership-boundary.md)
- [semantic concept](concepts/semantic-surface.md)
- [package entrypoint](../README.md)
- [repository docs index](../../../../docs/README.md)
