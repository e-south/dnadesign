"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_docs_contracts.py

Guardrails for compact, operator-first baserender documentation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def _pkg_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_docs_surface_stays_compact() -> None:
    root = _pkg_root()
    docs_md = sorted(str(p.relative_to(root)) for p in (root / "docs").rglob("*.md"))
    assert docs_md == [
        "docs/demos/workspaces.md",
        "docs/dev/journal.md",
        "docs/reference.md",
    ]


def test_readme_points_to_single_reference_and_examples() -> None:
    readme = (_pkg_root() / "README.md").read_text()
    assert "docs/reference.md" in readme
    assert "docs/demos/workspaces.md" in readme
    assert "docs/examples/*.yaml" in readme


def test_workspace_demo_guide_matches_output_contract() -> None:
    text = (_pkg_root() / "docs" / "demos" / "workspaces.md").read_text()
    assert "outputs/plots/" in text
    assert "run.emit_report: true" in text
