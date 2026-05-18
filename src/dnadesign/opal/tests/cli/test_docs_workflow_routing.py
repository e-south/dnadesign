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

    assert "### Start here" in docs_index
    assert "./workflows/usr-infer-x-active-learning.md" in docs_index
    assert "docs/index.md" in top_readme
    assert "docs/workflows/usr-infer-x-active-learning.md" in top_readme
    assert "infer has already written the chosen feature column into USR" in docs_index
    assert "../../../cluster/docs/workflows/exploratory-clustering.md" in _read(
        "src/dnadesign/opal/docs/workflows/usr-infer-x-active-learning.md"
    )


def test_opal_usr_infer_x_workflow_keeps_upstream_preconditions_explicit() -> None:
    workflow = _read("src/dnadesign/opal/docs/workflows/usr-infer-x-active-learning.md")

    assert "**Type:** workflow" in workflow
    assert "**Plane:** downstream-tool" in workflow
    assert "**Owner-boundary:** opal" in workflow
    assert "**Registry-id:** opal.downstream.usr-infer-x-active-learning" in workflow
    assert "**Execution-kind:** round-loop" in workflow
    assert "**Status-kind:** opal-campaign-state" in workflow
    assert "starts after infer write-back is already complete" in workflow
    assert "data.location.kind: usr" in workflow
    assert "x_column_name" in workflow
    assert "infer-derived `X` column" in workflow
    assert "../../../usr/docs/operations/promoter/characterization-feature-matrix.md" in workflow
