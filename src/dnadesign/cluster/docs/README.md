## Cluster docs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-20

Cluster is where you explore one chosen feature definition before committing to a supervised downstream branch. Use this page as the single route map for runnable workflows, command and artifact contracts, ownership boundaries, and adjacent handoffs back to USR, infer, or OPAL.

### Start here

1. If you do not yet have one explicit chosen feature definition, return to [promoter characterization feature matrix](../../usr/docs/operations/promoter-characterization-feature-matrix.md) for the repository's canonical infer-backed route.
2. If you already have a chosen feature definition and want the first runnable path, open [exploratory clustering workflow](workflows/exploratory-clustering.md).
3. If you need flag, workspace, preset, OPAL-join, or results semantics before running, open [Cluster CLI contracts](reference/cli-contracts.md).
4. If you need the public in-process automation boundary, open [`dnadesign.cluster.api`](../api.py).
5. If you need the downstream ownership split versus USR, infer, and OPAL, open [Cluster ownership boundary](concepts/ownership-boundary.md) and [Cluster semantic surface](concepts/semantic-surface.md).
6. If you are changing Cluster itself, open [Cluster verification contract](reference/verification.md).

### Route map

- [Exploratory clustering workflow](workflows/exploratory-clustering.md): first `fit -> umap -> analyze` path over one chosen feature definition.
- [Cluster CLI contracts](reference/cli-contracts.md): command, layout, OPAL-join, and troubleshooting contracts.
- [Cluster verification contract](reference/verification.md): deterministic package-local preflight/run/verify loop.
- [Cluster ownership boundary](concepts/ownership-boundary.md): ownership split versus USR, infer, and OPAL.
- [Cluster semantic surface](concepts/semantic-surface.md): package nouns and public runtime boundary rules.

### Adjacent handoffs

- [promoter characterization feature matrix](../../usr/docs/operations/promoter-characterization-feature-matrix.md): upstream durable data-plane handoff that creates one chosen feature definition.
- [infer docs](../../infer/docs/README.md): return here when the chosen feature definition does not exist yet.
- [USR dataset with infer-derived X -> OPAL active learning](../../opal/docs/workflows/usr-infer-x-active-learning.md): supervised downstream branch once exploratory clustering is no longer the next task.
- [Repository docs index](../../../../docs/README.md): repo-wide route map for upstream and downstream workflows.
