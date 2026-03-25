"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/tests/test_docs_contract.py

Documentation routing contracts for cluster cross-tool workflow entrypoints.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def test_cluster_readme_routes_back_to_root_docs_and_usr_feature_matrix_flow() -> None:
    readme = (_repo_root() / "src/dnadesign/cluster/README.md").read_text(encoding="utf-8")

    assert "## Documentation" in readme
    assert "docs/README.md" in readme
    assert "workspaces/README.md" in readme
    assert "docs/workflows/exploratory-clustering.md" in readme
    assert "docs/reference/cli-contracts.md" in readme
    assert "docs/reference/verification.md" in readme
    assert "docs/concepts/semantic-surface.md" in readme
    assert "../../../docs/README.md" in readme
    assert "Cluster is the exploratory downstream surface" in readme
    assert "## Ownership boundary" not in readme
    assert "## Start here" not in readme
    assert "## Task routes" not in readme
    assert "## CLI surface" not in readme
    assert "## Results and artifacts" not in readme


def test_cluster_docs_tree_exposes_workflow_reference_and_concept_surfaces() -> None:
    repo_root = _repo_root()
    workflow_index = (repo_root / "src/dnadesign/cluster/docs/README.md").read_text(encoding="utf-8")
    by_type = (repo_root / "src/dnadesign/cluster/docs/index.md").read_text(encoding="utf-8")
    workflow = (repo_root / "src/dnadesign/cluster/docs/workflows/exploratory-clustering.md").read_text(
        encoding="utf-8"
    )
    reference = (repo_root / "src/dnadesign/cluster/docs/reference/cli-contracts.md").read_text(encoding="utf-8")
    concept = (repo_root / "src/dnadesign/cluster/docs/concepts/ownership-boundary.md").read_text(encoding="utf-8")
    semantic = (repo_root / "src/dnadesign/cluster/docs/concepts/semantic-surface.md").read_text(encoding="utf-8")

    assert "### Start here" in workflow_index
    assert "### Route map" in workflow_index
    assert "### Adjacent handoffs" in workflow_index
    assert "workflows/exploratory-clustering.md" in workflow_index
    assert "reference/cli-contracts.md" in workflow_index
    assert "reference/verification.md" in workflow_index
    assert "../api.py" in workflow_index or "api.py" in workflow_index
    assert "concepts/ownership-boundary.md" in workflow_index
    assert "concepts/semantic-surface.md" in workflow_index
    assert "../../usr/docs/operations/promoter-characterization-feature-matrix.md" in workflow_index
    assert "../../infer/docs/README.md" in workflow_index
    assert "../../opal/docs/workflows/usr-infer-x-active-learning.md" in workflow_index

    assert "unified [Cluster docs](README.md) index" in by_type
    assert "Open the Cluster docs index" in by_type
    assert "exploratory clustering workflow" in by_type

    assert "### Preconditions" in workflow
    assert "**Type:** workflow" in workflow
    assert "**Plane:** downstream-tool" in workflow
    assert "**Owner-boundary:** cluster" in workflow
    assert "**Registry-id:** cluster.downstream.exploratory-clustering" in workflow
    assert "**Execution-kind:** exploratory" in workflow
    assert "**Status-kind:** cluster-run-index" in workflow
    assert "### First fit, UMAP, and analysis pass" in workflow
    assert "### Optional OPAL-join path" in workflow
    assert "--opal-campaign" in workflow
    assert "--opal-run latest|round:<n>|run_id:<rid>" in workflow
    assert "method.leiden.fine" in workflow
    assert "promoter_clusters_v1" in workflow
    assert "uv run cluster fit --workspace promoter_clusters_v1" in workflow
    assert "uv run cluster umap --workspace promoter_clusters_v1" in workflow
    assert "uv run cluster umap --workspace promoter_clusters_v1 --no-plots" in workflow
    assert "uv run cluster analyze --workspace promoter_clusters_v1" in workflow
    assert "uv run cluster sweep \\" in workflow
    assert "--results-root /tmp/cluster-promoter-demo" in workflow
    assert "analysis/<run-slug>/analysis.json" in workflow
    assert "sweep.json" in workflow
    assert "api.py" in workflow
    assert "cluster verification contract" in workflow
    assert "checked-in jobs" not in workflow

    assert "### Dataset and feature-column contract" in reference
    assert "### Public execution API" in reference
    assert "### Workspaces, presets, and results layout" in reference
    assert "### OPAL-join contract" in reference
    assert "### Results and artifacts" in reference
    assert "workspace config path fields" in reference
    assert "workspace.<section>.plot" in reference
    assert "umap.plot.enabled: false" in reference
    assert "`--no-plots`" in reference
    assert "`--workspace <workspace-id|path>`" in reference
    assert "`--results-root <path>`" in reference
    assert "`uv run cluster workspaces where`" in reference
    assert "`uv run cluster workspaces init --id my_run --root /tmp`" in reference
    assert "`uv run cluster workspaces list`" in reference
    assert "`uv run cluster workspaces show --help`" in reference
    assert "--method-param key=value" in reference
    assert "Legacy top-level fit method keys" in reference
    assert "`cluster sweep` is method-scoped and requires `--method`" in reference
    assert "fails fast instead of defaulting runtime state under `src/dnadesign/cluster/`" in reference
    assert "All attached overlay columns use one namespace contract" in reference
    assert "When `cluster analyze` omits `--out-dir`" in reference
    assert "`analysis/<run-slug>/analysis.json`" in reference
    assert "`../../api.py`" in reference
    assert "`run_fit()`" in reference
    assert "`run_fit_workspace()`" in reference
    assert "`sweep.json`" in reference
    assert "method signature" in reference
    assert "including any OPAL join inputs" in reference
    assert "immutable run slug" in reference or "immutable run" in reference

    workspaces = (repo_root / "src/dnadesign/cluster/workspaces/README.md").read_text(encoding="utf-8")
    assert "uv run cluster workspace list" in workspaces

    assert "### What cluster owns" in concept
    assert "### What cluster does not own" in concept
    assert "### Choose cluster when" in concept
    assert "### Do not stay in cluster when" in concept

    assert "### Core nouns" in semantic
    assert "`InputSource`" in semantic
    assert "`FeatureSpec`" in semantic
    assert "`AnalysisRequest`" in semantic
    assert "`WorkspaceConfig`" in semantic
    assert "`ClusterRun`" in semantic
    assert "`EmbeddingRun`" in semantic
    assert "`AnalysisRun`" in semantic
    assert "`SweepRun`" in semantic
    assert "`../../api.py`" in semantic
    assert "`../../contracts.py`" in semantic
    assert "`src/runtime_contracts.py`" in semantic
    assert "`src/workspaces/contracts.py`" in semantic
    assert "`src/workspaces/loader.py`" in semantic
    assert "`src/analysis/contracts.py`" in semantic
    assert "`src/runs/contracts.py`" in semantic
    assert "`src/runs/recorder.py`" in semantic

    verification = (repo_root / "src/dnadesign/cluster/docs/reference/verification.md").read_text(encoding="utf-8")

    assert "### Fast verify path" in verification
    assert "bash src/dnadesign/cluster/scripts/verify_cluster_contracts.sh" in verification
    assert "### Manual breakdown" in verification
    assert "mutating local workspace flow" in verification
    assert "umap.plot.enabled: false" in verification
    assert "uv run pytest -q \\" in verification
    assert "src/dnadesign/cluster/tests/test_source_tree_contracts.py" in verification
