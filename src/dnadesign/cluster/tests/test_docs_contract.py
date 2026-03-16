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

    assert "## Ownership boundary" in readme
    assert "## Start here" in readme
    assert "## Task routes" in readme
    assert "## Documentation map" in readme
    assert "../../../docs/README.md" in readme
    assert "docs/README.md" in readme
    assert "docs/index.md" in readme
    assert "docs/workflows/exploratory-clustering.md" in readme
    assert "docs/reference/cli-contracts.md" in readme
    assert "docs/reference/verification.md" in readme
    assert "docs/concepts/ownership-boundary.md" in readme
    assert "docs/concepts/semantic-surface.md" in readme
    assert "../usr/docs/operations/promoter-characterization-feature-matrix.md" in readme
    assert "Precondition: one explicit chosen feature definition is already present" in readme
    assert "If you do not yet have one explicit chosen feature definition" in readme
    assert "If you need supervised label/train/select rather than exploratory structure" in readme
    assert "../opal/docs/workflows/usr-infer-x-active-learning.md" in readme
    assert "uv run cluster fit --help" in readme
    assert "uv run cluster umap --help" in readme
    assert "uv run cluster analyze --help" in readme
    assert "DNADESIGN_CLUSTER_RESULTS_DIR" in readme
    assert "nearest project `cluster/results/`" in readme
    assert "will not write runtime state under the package tree" in readme


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
    assert "### Documentation by workflow" in workflow_index
    assert "### Documentation by type" in workflow_index
    assert "workflows/exploratory-clustering.md" in workflow_index
    assert "reference/cli-contracts.md" in workflow_index
    assert "reference/verification.md" in workflow_index
    assert "concepts/ownership-boundary.md" in workflow_index
    assert "concepts/semantic-surface.md" in workflow_index
    assert "../../usr/docs/operations/promoter-characterization-feature-matrix.md" in workflow_index
    assert "../../opal/docs/workflows/usr-infer-x-active-learning.md" in workflow_index

    assert "### Read order" in by_type
    assert "### Documentation by type" in by_type
    assert "workflows/exploratory-clustering.md" in by_type
    assert "reference/cli-contracts.md" in by_type
    assert "reference/verification.md" in by_type
    assert "concepts/ownership-boundary.md" in by_type
    assert "concepts/semantic-surface.md" in by_type

    assert "### Preconditions" in workflow
    assert "### First fit, UMAP, and analysis pass" in workflow
    assert "### Optional OPAL-join path" in workflow
    assert "--opal-campaign" in workflow
    assert "--opal-run latest|round:<n>|run_id:<rid>" in workflow
    assert "method.leiden.fine" in workflow
    assert "promoter_clusters_v1" in workflow
    assert "src/dnadesign/cluster/jobs/promoter_clusters_v1/fit.yaml" in workflow
    assert "uv run cluster fit \\" in workflow
    assert "uv run cluster umap \\" in workflow
    assert "uv run cluster analyze \\" in workflow
    assert "analysis/analysis.json" in workflow
    assert "cluster verification contract" in workflow

    assert "### Dataset and feature-column contract" in reference
    assert "### Jobs, presets, and results layout" in reference
    assert "### OPAL-join contract" in reference
    assert "### Results and artifacts" in reference
    assert "DNADESIGN_CLUSTER_RESULTS_DIR" in reference
    assert "DNADESIGN_CLUSTER_ROOT" in reference
    assert "checked-in job path fields" in reference
    assert "job.params.method_params" in reference
    assert "--method-param key=value" in reference
    assert "Legacy top-level fit method keys" in reference
    assert "`cluster sweep` is method-scoped and requires `--method`" in reference
    assert "fails fast instead of defaulting runtime state under `src/dnadesign/cluster/results`" in reference
    assert "When `cluster analyze` omits `--out-dir`" in reference
    assert "`analysis/analysis.json`" in reference
    assert "including any OPAL join inputs" in reference

    assert "### What cluster owns" in concept
    assert "### What cluster does not own" in concept
    assert "### Choose cluster when" in concept
    assert "### Do not stay in cluster when" in concept

    assert "### Core nouns" in semantic
    assert "`InputSource`" in semantic
    assert "`FeatureSpec`" in semantic
    assert "`AnalysisRequest`" in semantic
    assert "`ClusterRun`" in semantic
    assert "`EmbeddingRun`" in semantic
    assert "`AnalysisRun`" in semantic
    assert "`../../contracts.py`" in semantic
    assert "`src/runtime_contracts.py`" in semantic
    assert "`src/analysis/contracts.py`" in semantic
    assert "`src/runs/contracts.py`" in semantic
    assert "`src/runs/recorder.py`" in semantic

    verification = (repo_root / "src/dnadesign/cluster/docs/reference/verification.md").read_text(encoding="utf-8")

    assert "### Fast verify path" in verification
    assert "bash src/dnadesign/cluster/scripts/verify_cluster_contracts.sh" in verification
    assert "### Manual breakdown" in verification
    assert "uv run pytest -q \\" in verification
