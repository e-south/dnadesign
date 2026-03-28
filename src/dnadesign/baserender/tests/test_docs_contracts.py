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

from dnadesign.baserender import DENSEGEN_TFBS_REQUIRED_KEYS


def _pkg_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_docs_surface_stays_compact() -> None:
    root = _pkg_root()
    docs_md = sorted(str(p.relative_to(root)) for p in (root / "docs").rglob("*.md"))
    assert docs_md == [
        "docs/README.md",
        "docs/demos/workspaces.md",
        "docs/dev/journal.md",
        "docs/integrations/README.md",
        "docs/integrations/cruncher.md",
        "docs/integrations/densegen.md",
        "docs/integrations/yiu.md",
        "docs/reference.md",
    ]


def test_readme_points_to_single_reference_and_examples() -> None:
    readme = (_pkg_root() / "README.md").read_text()
    assert "docs/README.md" in readme
    assert "docs/reference.md" in readme
    assert "docs/demos/workspaces.md" in readme
    assert "docs/integrations/README.md" in readme
    assert "docs/examples" in readme


def test_baserender_docs_index_routes_to_reference_integrations_and_demos() -> None:
    text = (_pkg_root() / "docs" / "README.md").read_text()
    assert "### Start here" in text
    assert "### Documentation by type" in text
    assert "reference.md" in text
    assert "integrations/README.md" in text
    assert "demos/workspaces.md" in text
    assert "examples" in text


def test_readme_stays_tool_agnostic() -> None:
    readme = (_pkg_root() / "README.md").read_text()
    assert "densegen_notebook_render_contract" not in readme
    assert "demo_densegen_render" not in readme
    assert "demo_cruncher_render" not in readme


def test_workspace_demo_guide_matches_output_contract() -> None:
    text = (_pkg_root() / "docs" / "demos" / "workspaces.md").read_text()
    assert "outputs/plots/" in text
    assert "run.emit_report: true" in text
    assert "workspace init --root /path/to/workspaces demo_run" in text
    assert "inputs/input.parquet" in text
    assert "not BaseRender workspaces" in text


def test_densegen_integration_doc_declares_strict_tfbs_contract() -> None:
    text = (_pkg_root() / "docs" / "integrations" / "densegen.md").read_text()
    for key in DENSEGEN_TFBS_REQUIRED_KEYS:
        assert f"`{key}`" in text
    assert "Legacy TFBS keys (`tf`, `tfbs`, `stage_a_*`) are not accepted" in text
    assert "`on_invalid_row=error`" in text


def test_reference_and_cruncher_integration_docs_cover_cassette_json_contract_path() -> None:
    reference = (_pkg_root() / "docs" / "reference.md").read_text()
    cruncher = (_pkg_root() / "docs" / "integrations" / "cruncher.md").read_text()

    assert "`json`" in reference
    assert "`jsonl`" in reference
    assert "duplex_sequence_v1" in reference
    assert "hairpin_topology_v1" in reference
    assert "linear_duplex.v1.json" in cruncher
    assert "top_hits.linear_duplex.v1.jsonl" in cruncher
    assert "duplex_sequence_v1" in cruncher
