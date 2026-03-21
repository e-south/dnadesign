![cluster banner](assets/cluster-banner.svg)

Cluster is the exploratory downstream surface for exploratory clustering, UMAP visualization, and related summaries after one feature column or exported matrix already exists. It works from a chosen feature matrix in a USR dataset or a CSV/Parquet file and records reusable outputs in a workspace-scoped artifact root. It does not generate upstream features.

## Documentation

- [Cluster docs](docs/README.md): first runs, concepts, references, and cross-tool handoffs.
- [Workspaces guide](workspaces/README.md): inspect packaged workspaces with `uv run cluster workspace list` and scaffold local runs.
- [Exploratory clustering workflow](docs/workflows/exploratory-clustering.md): first runnable `fit -> umap -> analyze` path once one feature column or exported matrix exists.
- [Cluster CLI contracts](docs/reference/cli-contracts.md): commands, workspace or preset layout, results policy, and OPAL join contract.
- [Cluster verification contract](docs/reference/verification.md): deterministic preflight/run/verify path for package changes.
- [Cluster semantic surface](docs/concepts/semantic-surface.md): package nouns, runtime boundaries, and public API surface.
- [Repository docs index](../../../docs/README.md): repo-wide docs index for upstream and downstream cross-tool workflows.
