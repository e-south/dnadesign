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
    assert "../usr/docs/operations/promoter-characterization-feature-matrix.md" in readme
    assert "Precondition: one explicit `infer__...` column is already present and chosen as `X`" in readme
    assert "If you do not yet have one explicit `infer__...` column chosen as `X`" in readme
    assert "If you need supervised label/train/select instead of exploratory clustering" in readme
    assert "../opal/docs/workflows/usr-infer-x-active-learning.md" in readme
    assert "cluster fit" in readme
    assert "cluster umap" in readme
    assert "cluster analyze" in readme
    assert "--opal-campaign" in readme
    assert "--opal-run" in readme
