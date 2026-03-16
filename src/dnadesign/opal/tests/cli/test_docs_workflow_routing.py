"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/cli/test_docs_workflow_routing.py

Documentation routing contracts for OPAL cross-tool workflow ownership.

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


def _read(rel_path: str) -> str:
    return (_repo_root() / rel_path).read_text(encoding="utf-8")


def test_opal_docs_index_routes_to_usr_infer_x_workflow() -> None:
    docs_index = _read("src/dnadesign/opal/docs/index.md")
    top_readme = _read("src/dnadesign/opal/README.md")

    assert "./workflows/usr-infer-x-active-learning.md" in docs_index
    assert "./docs/workflows/usr-infer-x-active-learning.md" in top_readme


def test_opal_usr_infer_x_workflow_keeps_upstream_preconditions_explicit() -> None:
    workflow = _read("src/dnadesign/opal/docs/workflows/usr-infer-x-active-learning.md")

    assert "starts after infer write-back is already complete" in workflow
    assert "data.location.kind: usr" in workflow
    assert "x_column_name" in workflow
    assert "infer-derived `X` column" in workflow
    assert "../../../usr/docs/operations/promoter-characterization-feature-matrix.md" in workflow
