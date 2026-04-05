"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/docs/test_yiu_docs_routing.py

Routing contracts for payload-centric YIU docs surfaces.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_top_level_docs_route_readers_to_yiu_surfaces() -> None:
    package_readme = _read("README.md")
    docs_readme = _read("docs/README.md")
    docs_index = _read("docs/index.md")

    assert "payload-centric YIU" in package_readme
    assert "docs/demos/demo_yiu_workspace.md" in package_readme
    assert "docs/guides/yiu_workflow.md" in package_readme

    for content in (docs_readme, docs_index):
        assert "Payload-Centric YIU Workflows" in content
        assert "demos/demo_yiu_workspace.md" in content
        assert "guides/yiu_workflow.md" in content
        assert "reference/yiu_spec.md" in content
        assert "reference/yiu_artifacts.md" in content
        assert "reference/yiu_visual_system.md" in content
        assert "yiu init-workspace|validate|render|show" in content
        assert "trace|solve" not in content


def test_cli_reference_lists_public_yiu_surface() -> None:
    cli_ref = _read("docs/reference/cli.md")

    assert "YIU workflows" in cli_ref
    assert "cruncher yiu init-workspace" in cli_ref
    assert "cruncher yiu validate" in cli_ref
    assert "cruncher yiu render" in cli_ref
    assert "cruncher yiu show" in cli_ref
    assert "split_yiu_payload_rendering_v4" in cli_ref
    assert "Treat the bundle directory as the source of truth" in cli_ref


def test_yiu_workflow_routes_to_contract_pages() -> None:
    guide = _read("docs/guides/yiu_workflow.md")

    assert "This page is the operator route map" in guide
    assert "### Documentation ownership" in guide
    assert "[YIU Spec Reference](../reference/yiu_spec.md)" in guide
    assert "[YIU Artifacts](../reference/yiu_artifacts.md)" in guide
    assert "[YIU Visual System](../reference/yiu_visual_system.md)" in guide
    assert "[Cruncher architecture](../reference/architecture.md)" in guide
    assert "Ambiguous or missing sources fail fast." in guide
    assert "Cross-tool integrations should not import `dnadesign.baserender.src.*`." in guide
