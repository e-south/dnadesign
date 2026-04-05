"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/docs/test_docs_integrity_tools.py

Regression contracts for Cruncher docs integrity tooling.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.devtools.docs_lint import _lint_runbook_coupling, _lint_schema_mentions

ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = ROOT / "docs"


def _relative_doc_path(path: Path) -> str:
    return path.relative_to(DOCS_ROOT).as_posix()


def test_docs_lint_accepts_cassette_family_schema_examples() -> None:
    issues = _lint_schema_mentions()
    target_paths = {
        "guides/cassette_workflow.md",
        "guides/cassette_solve_workflow.md",
        "reference/cassette_spec.md",
        "reference/cassette_solve_spec.md",
        "reference/nickase_catalog.md",
    }

    relevant = [issue for issue in issues if _relative_doc_path(issue.path) in target_paths]

    assert relevant == []


def test_demo_cassette_workspace_does_not_require_checked_in_runbook_root() -> None:
    issues = _lint_runbook_coupling()
    relevant = [issue for issue in issues if _relative_doc_path(issue.path) == "demos/demo_cassette_workspace.md"]

    assert relevant == []
