"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/test_docs_contract.py

Documentation placement contracts for construct cross-tool workflow routing.

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


def test_construct_docs_route_to_shared_source_of_truth_runbook() -> None:
    top_readme = _read("src/dnadesign/construct/README.md")
    readme = _read("src/dnadesign/construct/docs/README.md")
    index_doc = _read("src/dnadesign/construct/docs/index.md")
    config_doc = _read("src/dnadesign/construct/docs/reference/config.md")
    outputs = _read("src/dnadesign/construct/docs/reference/outputs.md")
    template_contexts = _read("src/dnadesign/construct/docs/reference/template-contexts.md")
    workspaces = _read("src/dnadesign/construct/workspaces/README.md")
    source_of_truth_workspace = _read(
        "src/dnadesign/construct/workspaces/demo_promoter_swap_pdual10_source_of_truth/README.md"
    )

    token = "../../usr/docs/operations/construct-infer-source-of-truth-runbook.md"
    multi_source_token = "../../usr/docs/operations/multi-source-source-of-truth-assembly.md"
    feature_matrix_token = "../../usr/docs/operations/promoter-characterization-feature-matrix.md"
    assert token in readme
    assert multi_source_token in readme
    assert feature_matrix_token in readme
    assert "## Documentation" in top_readme
    assert "docs/README.md" in top_readme
    assert "docs/getting-started.md" in top_readme
    assert "workspaces/README.md" in top_readme
    assert "docs/reference/template-contexts.md" in top_readme
    assert "../../../docs/README.md" in top_readme
    assert "## Start here" not in top_readme
    assert "## Boundary reminder" not in top_readme
    assert "Construct takes focal DNA parts" in top_readme
    assert "reference/template-contexts.md" in readme
    assert "template-contexts.md" in config_doc
    assert "../../../usr/docs/operations/construct-infer-source-of-truth-runbook.md" in outputs
    assert "construct__anchor_start" in outputs
    assert "construct__anchor_start" in template_contexts
    assert "### Key docs" in readme
    assert "### Boundary reminders" in readme
    assert "demo_promoter_swap_pdual10_source_of_truth" in readme
    assert "demo_promoter_swap_pdual10_source_of_truth" in workspaces
    assert "uv run construct workspace list" in workspaces
    assert "downstream consumers" in source_of_truth_workspace
    assert "promoter-characterization-feature-matrix.md" in source_of_truth_workspace
    assert "unified [Construct docs](README.md) index" in index_doc
    assert "Open the Construct docs index" in index_doc
