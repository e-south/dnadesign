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
    assert "docs/concepts/ownership-boundary.md" in readme
    assert "../usr/docs/operations/promoter-characterization-feature-matrix.md" in readme
    assert "Precondition: one explicit `infer__...` column is already present and chosen as `X`" in readme
    assert "If you do not yet have one explicit `infer__...` column chosen as `X`" in readme
    assert "If you need supervised label/train/select rather than exploratory structure" in readme
    assert "../opal/docs/workflows/usr-infer-x-active-learning.md" in readme
    assert "uv run cluster fit --help" in readme
    assert "uv run cluster umap --help" in readme
    assert "uv run cluster analyze --help" in readme


def test_cluster_docs_tree_exposes_workflow_reference_and_concept_surfaces() -> None:
    repo_root = _repo_root()
    workflow_index = (repo_root / "src/dnadesign/cluster/docs/README.md").read_text(encoding="utf-8")
    by_type = (repo_root / "src/dnadesign/cluster/docs/index.md").read_text(encoding="utf-8")
    workflow = (repo_root / "src/dnadesign/cluster/docs/workflows/exploratory-clustering.md").read_text(
        encoding="utf-8"
    )
    reference = (repo_root / "src/dnadesign/cluster/docs/reference/cli-contracts.md").read_text(encoding="utf-8")
    concept = (repo_root / "src/dnadesign/cluster/docs/concepts/ownership-boundary.md").read_text(encoding="utf-8")

    assert "### Start here" in workflow_index
    assert "### Documentation by workflow" in workflow_index
    assert "### Documentation by type" in workflow_index
    assert "workflows/exploratory-clustering.md" in workflow_index
    assert "reference/cli-contracts.md" in workflow_index
    assert "concepts/ownership-boundary.md" in workflow_index
    assert "../../usr/docs/operations/promoter-characterization-feature-matrix.md" in workflow_index
    assert "../../opal/docs/workflows/usr-infer-x-active-learning.md" in workflow_index

    assert "### Read order" in by_type
    assert "### Documentation by type" in by_type
    assert "workflows/exploratory-clustering.md" in by_type
    assert "reference/cli-contracts.md" in by_type
    assert "concepts/ownership-boundary.md" in by_type

    assert "### Preconditions" in workflow
    assert "### First fit, UMAP, and analysis pass" in workflow
    assert "### Optional OPAL-join path" in workflow
    assert "--opal-campaign" in workflow
    assert "--opal-run latest|round:<n>|run_id:<rid>" in workflow
    assert "uv run cluster fit \\" in workflow
    assert "uv run cluster umap \\" in workflow
    assert "uv run cluster analyze \\" in workflow

    assert "### Dataset and feature-column contract" in reference
    assert "### Jobs, presets, and results layout" in reference
    assert "### OPAL-join contract" in reference
    assert "### Results and artifacts" in reference
    assert "DNADESIGN_CLUSTER_RESULTS_DIR" in reference

    assert "### What cluster owns" in concept
    assert "### What cluster does not own" in concept
    assert "### Choose cluster when" in concept
    assert "### Do not stay in cluster when" in concept
