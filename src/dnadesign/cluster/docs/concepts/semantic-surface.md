## Cluster semantic surface

**Owner:** dnadesign-maintainers
**Last verified:** 2026-06-14

Use this page when you need the package nouns and boundaries before changing code or contracts.

### Core nouns

- `InputSource`: a USR dataset or a CSV/Parquet file that provides row-aligned records.
- `FeatureSpec`: the explicit feature definition used to build the clustering matrix, either one vector column or one numeric column set.
- `AnalysisRequest`: the resolved downstream analysis execution request after CLI/workspace/preset precedence has been applied.
- `WorkspaceConfig`: the canonical reusable machine config surface for `fit`, `umap`, and `analyze`.
- `ClusteringMethod`: the algorithm identity, such as `leiden` or `kmeans`.
- `MethodConfig`: the parsed method-specific parameters for the chosen clustering method.
- `ClusterRun`: the immutable fit artifact produced from one `InputSource`, one `FeatureSpec`, and one `MethodConfig`, stored under one stable alias and one unique run slug.
- `ClusterAssignment`: the row-aligned labels and metadata attached back to a dataset or file.
- `EmbeddingRun`: a downstream embedding artifact, such as UMAP coordinates, derived from a chosen feature matrix or fitted clustering context.
- `AnalysisRun`: downstream summaries and plots derived from an existing cluster assignment.
- `SweepRun`: a method-scoped resolution sweep artifact recorded under the workspace run ledger.

### Boundary rules

- `cluster` consumes tabular records and explicit feature definitions; it does not generate upstream feature columns.
- The package-level contract is capability-based, not workflow-specific. Infer-derived features are one common route, not the only valid input surface.
- Method-specific knobs belong to `MethodConfig`, not to the package-wide ontology.
- Public package-to-package automation belongs on [`../../api.py`](../../api.py), not internal CLI helpers. That API exposes both ad hoc execution functions and workspace helpers, and both call the shared cluster runtime directly instead of shelling back through the CLI.
- Runtime artifacts belong in a workspace `outputs/cluster/` root or an explicit standalone results root, never under the built-in package tree.
- Stable alias roots group related artifacts, while immutable run slugs prevent ledger rows from pointing at overwritten files.
- `EmbeddingRun` and `AnalysisRun` consume `ClusterRun` outputs; they do not redefine the fit contract.
- External integrations should use the public [`../../contracts.py`](../../contracts.py), [`../../api.py`](../../api.py), or the CLI, not internal `src.*` modules.

### Current code mapping

- Public contract surface: [`../../contracts.py`](../../contracts.py)
- Public execution surface: [`../../api.py`](../../api.py)
- `InputSource`, `FeatureSpec`, and `MethodConfig`: [`src/runtime_contracts.py`](../../src/runtime_contracts.py)
- `WorkspaceConfig`: [`src/workspaces/contracts.py`](../../src/workspaces/contracts.py) and [`src/workspaces/loader.py`](../../src/workspaces/loader.py)
- `AnalysisRequest`: [`src/analysis/contracts.py`](../../src/analysis/contracts.py)
- `InputSource` loading path: [`io/detect.py`](../../src/io/detect.py) and [`io/read.py`](../../src/io/read.py)
- `ClusteringMethod`: [`src/methods/`](../../src/methods)
- `ClusterRun`, `EmbeddingRun`, `AnalysisRun`, and `SweepRun`: [`src/runs/contracts.py`](../../src/runs/contracts.py)
- Run-artifact persistence: [`src/runs/recorder.py`](../../src/runs/recorder.py)
- `ClusterAssignment`: [`src/cli/app.py`](../../src/cli/app.py) fit attachment flow
- UMAP execution and plotting: [`src/umap/`](../../src/umap)
- Analysis computations: [`src/analysis/`](../../src/analysis)

### Related docs

- [cluster ownership boundary](ownership-boundary.md)
- [cluster CLI contracts](../reference/cli-contracts.md)
- [cluster verification contract](../reference/verification.md)
- [exploratory clustering workflow](../workflows/exploratory-clustering.md)
