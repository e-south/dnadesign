## Cluster docs

**Owner:** dnadesign-maintainers
**Last verified:** 2026-07-14

Cluster is where you explore one feature column or exported matrix before committing to a supervised downstream branch. Use this page to choose the next workflow, reference page, or handoff back to USR, infer, or OPAL.

### Start here

1. If you do not yet have the feature column or exported matrix you want to analyze, return to [promoter characterization feature matrix](../../usr/docs/operations/promoter/characterization-feature-matrix.md) for the shared infer-backed workflow.
2. If you already have that feature column or matrix and want the first runnable path, open [exploratory clustering workflow](workflows/exploratory-clustering.md).
3. If you need flag, workspace, preset, OPAL-join, or results semantics before running, open [Cluster CLI contracts](reference/cli-contracts.md).
4. If you need the public in-process automation boundary, open [`dnadesign.cluster.api`](../api.py).
5. If you need the downstream ownership split versus USR, infer, and OPAL, open [Cluster ownership boundary](concepts/ownership-boundary.md) and [Cluster semantic surface](concepts/semantic-surface.md).
6. If you are changing Cluster itself, open [Cluster verification contract](reference/verification.md).

### Route map

- [Exploratory clustering workflow](workflows/exploratory-clustering.md): first `fit -> umap -> analyze` path over one chosen feature column or exported matrix.
- [Cluster CLI contracts](reference/cli-contracts.md): command, layout, OPAL-join, and troubleshooting contracts.
- [Cluster verification contract](reference/verification.md): deterministic package-local preflight/run/verify loop.
- [Cluster ownership boundary](concepts/ownership-boundary.md): ownership split versus USR, infer, and OPAL.
- [Cluster semantic surface](concepts/semantic-surface.md): package nouns and public runtime boundary rules.

### Adjacent handoffs

- [promoter characterization feature matrix](../../usr/docs/operations/promoter/characterization-feature-matrix.md): upstream feature-matrix workflow that creates the feature column or exported matrix Cluster reads.
- [infer docs](../../infer/docs/README.md): return here when the feature column or exported matrix does not exist yet.
- [USR dataset with infer-derived X -> OPAL active learning](../../opal/docs/workflows/usr-infer-x-active-learning.md): supervised downstream branch once exploratory clustering is no longer the next task.
- [Repository docs index](../../../../docs/README.md): repo-wide docs index for upstream and downstream workflows.
