"""
Docs contracts for preserved-site Snapback workflow boundaries.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_cli_and_architecture_docs_capture_preserved_site_snapback_boundary() -> None:
    cli_ref = _read("docs/reference/cli.md")
    guide = _read("docs/guides/snapback_workflow.md")
    architecture_ref = _read("docs/reference/architecture.md")

    assert "cruncher snapback validate" in cli_ref
    assert "cruncher snapback design" in cli_ref
    assert "cruncher snapback solve" in cli_ref
    assert "cruncher snapback show" in cli_ref

    assert "single_nick_snapback_v2" in guide
    assert "single_nick_snapback_solve_v3" in guide
    assert "show" in guide

    assert "app/snapback_cli_requests.py" in architecture_ref
    assert "app/snapback_catalogs.py" in architecture_ref
    assert "app/snapback_workflow.py" in architecture_ref
    assert "app/snapback_show_{load,validate,present}.py" in architecture_ref
    assert "app/snapback_solve_workflow.py" in architecture_ref
    assert "app/snapback_solve_{snapshot,materialize,reporting}.py" in architecture_ref
    assert "snapback/preserved_search/" in architecture_ref
    assert "snapback/target_search.py" in architecture_ref
