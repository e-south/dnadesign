## cluster semantic surface

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-16

Use this page when you need the package nouns and boundaries before changing code or contracts.

### Core nouns

- `InputSource`: a USR dataset or a CSV/Parquet file that provides row-aligned records.
- `FeatureSpec`: the explicit feature definition used to build the clustering matrix, either one vector column or one numeric column set.
- `AnalysisRequest`: the resolved downstream analysis execution request after CLI/job/preset precedence has been applied.
- `ClusteringMethod`: the algorithm identity, such as `leiden`.
- `MethodConfig`: the parsed method-specific parameters for the chosen clustering method.
- `ClusterRun`: the immutable fit artifact produced from one `InputSource`, one `FeatureSpec`, and one `MethodConfig`.
- `ClusterAssignment`: the row-aligned labels and metadata attached back to a dataset or file.
- `EmbeddingRun`: a downstream embedding artifact, such as UMAP coordinates, derived from a chosen feature matrix or fitted clustering context.
- `AnalysisRun`: downstream summaries and plots derived from an existing cluster assignment.

### Boundary rules

- `cluster` consumes tabular records and explicit feature definitions; it does not generate upstream feature columns.
- The package-level contract is capability-based, not workflow-specific. Infer-derived features are one common route, not the only valid input surface.
- Method-specific knobs belong to `MethodConfig`, not to the package-wide ontology.
- Runtime artifacts belong in an explicit results root or a writable workspace `cluster/` directory, never under the built-in package tree.
- `EmbeddingRun` and `AnalysisRun` consume `ClusterRun` outputs; they do not redefine the fit contract.

### Current code mapping

- Public contract surface: [`../../contracts.py`](../../contracts.py)
- `InputSource`, `FeatureSpec`, and `MethodConfig`: [`src/runtime_contracts.py`](../../src/runtime_contracts.py)
- `AnalysisRequest`: [`src/analysis/contracts.py`](../../src/analysis/contracts.py)
- `InputSource` loading path: [`io/detect.py`](../../src/io/detect.py) and [`io/read.py`](../../src/io/read.py)
- `ClusteringMethod`: [`src/methods/`](../../src/methods)
- `ClusterRun`, `EmbeddingRun`, and `AnalysisRun`: [`src/runs/contracts.py`](../../src/runs/contracts.py)
- Run-artifact persistence: [`src/runs/recorder.py`](../../src/runs/recorder.py)
- `ClusterAssignment`: [`src/cli/app.py`](../../src/cli/app.py) fit attachment flow
- UMAP execution and plotting: [`src/umap/`](../../src/umap)
- Analysis computations: [`src/analysis/`](../../src/analysis)

### Related docs

- [cluster ownership boundary](ownership-boundary.md)
- [cluster CLI contracts](../reference/cli-contracts.md)
- [cluster verification contract](../reference/verification.md)
- [exploratory clustering workflow](../workflows/exploratory-clustering.md)
