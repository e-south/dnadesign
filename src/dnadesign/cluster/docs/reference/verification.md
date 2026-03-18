## cluster verification contract

**Owner:** dnadesign-maintainers
**Last verified:** 2026-03-18

Use this page when you are changing `cluster` code, docs, or checked-in workspaces/presets and want the smallest deterministic verification path.

### Start here

1. Run the fast verification script first.
2. Use the manual breakdown only when you need to widen or isolate one failure.
3. If you changed method math or broader dataflow behavior, widen beyond this package-local surface after the fast path is green.

### Fast verify path

```bash
bash src/dnadesign/cluster/scripts/verify_cluster_contracts.sh
```

### What the fast path checks

- runtime contracts and run-artifact typing
- source-tree and workspace information-architecture contracts
- docs progressive-disclosure routes and reference links
- public import boundary plus CLI bootstrap side effects
- public ad hoc and workspace execution APIs running through the shared cluster runtime
- targeted `ruff` and `py_compile` checks
- primary CLI help surfaces for `fit`, `umap`, `analyze`, and `sweep`
- workspace lifecycle discovery via `cluster workspaces where`
- one read-only workspace-scoped run-ledger command against a checked-in workspace
- one real mutating local workspace flow that runs `fit -> umap -> analyze -> sweep` on a tiny generated dataset under `/tmp`
- that same mutating workspace flow proves `umap.plot.enabled: false` still records coords cleanly without mandatory PNG rendering
- one public-API workspace flow that runs the same sequence without shelling back through the CLI
- one public-API ad hoc flow that runs the same sequence against an explicit results root

### Manual breakdown

#### Preflight

```bash
git status --short src/dnadesign/cluster
uv run cluster --help
uv run cluster workspaces where
uv run cluster runs list --workspace promoter_clusters_v1
```

#### Mutating workspace smoke path

Use the fast verify script for the authoritative tiny-workspace mutation path. It creates a temporary local workspace, runs `fit`, `umap`, `analyze`, and `sweep`, and asserts that the workspace-scoped artifact root records each run kind.

#### Run

```bash
uv run pytest -q \
  src/dnadesign/cluster/tests/test_runtime_contracts.py \
  src/dnadesign/cluster/tests/test_source_tree_contracts.py \
  src/dnadesign/cluster/tests/test_docs_contract.py \
  src/dnadesign/cluster/tests/test_cluster_public_import_boundary.py \
  src/dnadesign/cluster/src/cli/tests/test_import_side_effects.py
```

#### Verify

```bash
uv run ruff check \
  src/dnadesign/cluster/api.py \
  src/dnadesign/cluster/__init__.py \
  src/dnadesign/cluster/contracts.py \
  src/dnadesign/cluster/src/analysis/contracts.py \
  src/dnadesign/cluster/src/analysis/numeric_per_cluster.py \
  src/dnadesign/cluster/src/cli/app.py \
  src/dnadesign/cluster/src/cli/commands.py \
  src/dnadesign/cluster/src/cli/commands_analysis.py \
  src/dnadesign/cluster/src/cli/commands_fit.py \
  src/dnadesign/cluster/src/cli/commands_table.py \
  src/dnadesign/cluster/src/cli/commands_umap.py \
  src/dnadesign/cluster/src/cli/resolution.py \
  src/dnadesign/cluster/src/cli/subapps.py \
  src/dnadesign/cluster/src/cli/umap_resolution.py \
  src/dnadesign/cluster/src/execution.py \
  src/dnadesign/cluster/src/execution_analysis.py \
  src/dnadesign/cluster/src/execution_analysis_support.py \
  src/dnadesign/cluster/src/execution_fit.py \
  src/dnadesign/cluster/src/execution_fit_support.py \
  src/dnadesign/cluster/src/execution_sweep.py \
  src/dnadesign/cluster/src/execution_support.py \
  src/dnadesign/cluster/src/execution_table.py \
  src/dnadesign/cluster/src/execution_umap.py \
  src/dnadesign/cluster/src/io/parquet_attach.py \
  src/dnadesign/cluster/src/methods/__init__.py \
  src/dnadesign/cluster/src/methods/kmeans.py \
  src/dnadesign/cluster/src/methods/params.py \
  src/dnadesign/cluster/src/methods/registry.py \
  src/dnadesign/cluster/src/presets/runtime.py \
  src/dnadesign/cluster/src/presets/schema.py \
  src/dnadesign/cluster/src/runs/contracts.py \
  src/dnadesign/cluster/src/runs/index.py \
  src/dnadesign/cluster/src/runs/index_store.py \
  src/dnadesign/cluster/src/umap/contracts.py \
  src/dnadesign/cluster/src/umap/frame.py \
  src/dnadesign/cluster/src/umap/hues.py \
  src/dnadesign/cluster/src/umap/overlays.py \
  src/dnadesign/cluster/src/umap/plot.py \
  src/dnadesign/cluster/src/umap/requests.py \
  src/dnadesign/cluster/src/util/meta.py \
  src/dnadesign/cluster/src/workspaces/__init__.py \
  src/dnadesign/cluster/src/workspaces/errors.py \
  src/dnadesign/cluster/src/workspaces/loader.py \
  src/dnadesign/cluster/src/workspaces/paths.py \
  src/dnadesign/cluster/src/workspaces/schema.py \
  src/dnadesign/cluster/tests/test_runtime_contracts.py \
  src/dnadesign/cluster/tests/test_docs_contract.py \
  src/dnadesign/cluster/tests/test_cluster_public_import_boundary.py \
  src/dnadesign/cluster/src/cli/tests/test_import_side_effects.py

uv run ruff format --check \
  src/dnadesign/cluster/api.py \
  src/dnadesign/cluster/__init__.py \
  src/dnadesign/cluster/contracts.py \
  src/dnadesign/cluster/src/analysis/contracts.py \
  src/dnadesign/cluster/src/analysis/numeric_per_cluster.py \
  src/dnadesign/cluster/src/cli/app.py \
  src/dnadesign/cluster/src/cli/commands.py \
  src/dnadesign/cluster/src/cli/commands_analysis.py \
  src/dnadesign/cluster/src/cli/commands_fit.py \
  src/dnadesign/cluster/src/cli/commands_table.py \
  src/dnadesign/cluster/src/cli/commands_umap.py \
  src/dnadesign/cluster/src/cli/resolution.py \
  src/dnadesign/cluster/src/cli/subapps.py \
  src/dnadesign/cluster/src/cli/umap_resolution.py \
  src/dnadesign/cluster/src/execution.py \
  src/dnadesign/cluster/src/execution_analysis.py \
  src/dnadesign/cluster/src/execution_analysis_support.py \
  src/dnadesign/cluster/src/execution_fit.py \
  src/dnadesign/cluster/src/execution_fit_support.py \
  src/dnadesign/cluster/src/execution_sweep.py \
  src/dnadesign/cluster/src/execution_support.py \
  src/dnadesign/cluster/src/execution_table.py \
  src/dnadesign/cluster/src/execution_umap.py \
  src/dnadesign/cluster/src/io/parquet_attach.py \
  src/dnadesign/cluster/src/methods/__init__.py \
  src/dnadesign/cluster/src/methods/kmeans.py \
  src/dnadesign/cluster/src/methods/params.py \
  src/dnadesign/cluster/src/methods/registry.py \
  src/dnadesign/cluster/src/presets/runtime.py \
  src/dnadesign/cluster/src/presets/schema.py \
  src/dnadesign/cluster/src/runs/contracts.py \
  src/dnadesign/cluster/src/runs/index.py \
  src/dnadesign/cluster/src/runs/index_store.py \
  src/dnadesign/cluster/src/umap/contracts.py \
  src/dnadesign/cluster/src/umap/frame.py \
  src/dnadesign/cluster/src/umap/hues.py \
  src/dnadesign/cluster/src/umap/overlays.py \
  src/dnadesign/cluster/src/umap/plot.py \
  src/dnadesign/cluster/src/umap/requests.py \
  src/dnadesign/cluster/src/util/meta.py \
  src/dnadesign/cluster/src/workspaces/__init__.py \
  src/dnadesign/cluster/src/workspaces/errors.py \
  src/dnadesign/cluster/src/workspaces/loader.py \
  src/dnadesign/cluster/src/workspaces/paths.py \
  src/dnadesign/cluster/src/workspaces/schema.py \
  src/dnadesign/cluster/tests/test_runtime_contracts.py \
  src/dnadesign/cluster/tests/test_docs_contract.py \
  src/dnadesign/cluster/tests/test_cluster_public_import_boundary.py \
  src/dnadesign/cluster/src/cli/tests/test_import_side_effects.py

uv run python -m py_compile \
  src/dnadesign/cluster/api.py \
  src/dnadesign/cluster/__init__.py \
  src/dnadesign/cluster/contracts.py \
  src/dnadesign/cluster/src/analysis/contracts.py \
  src/dnadesign/cluster/src/analysis/numeric_per_cluster.py \
  src/dnadesign/cluster/src/cli/app.py \
  src/dnadesign/cluster/src/cli/commands.py \
  src/dnadesign/cluster/src/cli/commands_analysis.py \
  src/dnadesign/cluster/src/cli/commands_fit.py \
  src/dnadesign/cluster/src/cli/commands_table.py \
  src/dnadesign/cluster/src/cli/commands_umap.py \
  src/dnadesign/cluster/src/cli/resolution.py \
  src/dnadesign/cluster/src/cli/subapps.py \
  src/dnadesign/cluster/src/cli/umap_resolution.py \
  src/dnadesign/cluster/src/execution.py \
  src/dnadesign/cluster/src/execution_analysis.py \
  src/dnadesign/cluster/src/execution_analysis_support.py \
  src/dnadesign/cluster/src/execution_fit.py \
  src/dnadesign/cluster/src/execution_fit_support.py \
  src/dnadesign/cluster/src/execution_sweep.py \
  src/dnadesign/cluster/src/execution_support.py \
  src/dnadesign/cluster/src/execution_table.py \
  src/dnadesign/cluster/src/execution_umap.py \
  src/dnadesign/cluster/src/io/parquet_attach.py \
  src/dnadesign/cluster/src/methods/__init__.py \
  src/dnadesign/cluster/src/methods/kmeans.py \
  src/dnadesign/cluster/src/methods/params.py \
  src/dnadesign/cluster/src/methods/registry.py \
  src/dnadesign/cluster/src/presets/runtime.py \
  src/dnadesign/cluster/src/presets/schema.py \
  src/dnadesign/cluster/src/runs/contracts.py \
  src/dnadesign/cluster/src/runs/index.py \
  src/dnadesign/cluster/src/runs/index_store.py \
  src/dnadesign/cluster/src/umap/contracts.py \
  src/dnadesign/cluster/src/umap/frame.py \
  src/dnadesign/cluster/src/umap/hues.py \
  src/dnadesign/cluster/src/umap/overlays.py \
  src/dnadesign/cluster/src/umap/plot.py \
  src/dnadesign/cluster/src/umap/requests.py \
  src/dnadesign/cluster/src/util/meta.py \
  src/dnadesign/cluster/src/workspaces/__init__.py \
  src/dnadesign/cluster/src/workspaces/errors.py \
  src/dnadesign/cluster/src/workspaces/loader.py \
  src/dnadesign/cluster/src/workspaces/paths.py \
  src/dnadesign/cluster/src/workspaces/schema.py \
  src/dnadesign/cluster/tests/test_runtime_contracts.py \
  src/dnadesign/cluster/tests/test_docs_contract.py \
  src/dnadesign/cluster/tests/test_cluster_public_import_boundary.py \
  src/dnadesign/cluster/src/cli/tests/test_import_side_effects.py
```

### When to widen

- If you changed clustering method implementation, rerun the broader `cluster` tests beyond the contract set above.
- If you changed cross-tool docs or upstream/downstream handoffs, rerun the repository docs checks as well.
- If you changed runtime artifact schemas, inspect representative `fits/<run-slug>/run.json`, `umap/<run-slug>/umap.json`, or `analysis/<run-slug>/analysis.json` outputs in a writable results root after the fast path.

### Related docs

- [cluster CLI contracts](cli-contracts.md)
- [cluster semantic surface](../concepts/semantic-surface.md)
- [exploratory clustering workflow](../workflows/exploratory-clustering.md)
