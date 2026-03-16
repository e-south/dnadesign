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

    assert "../../../docs/README.md" in readme
    assert "../usr/docs/operations/promoter-characterization-feature-matrix.md" in readme
    assert "--opal-campaign" in readme
    assert "--opal-run" in readme
