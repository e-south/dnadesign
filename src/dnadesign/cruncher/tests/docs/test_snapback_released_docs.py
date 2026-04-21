"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/docs/test_snapback_released_docs.py

Docs contracts for the released-product snapback lane.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_docs_hubs_route_to_released_snapback_surfaces() -> None:
    docs_readme = _read("docs/README.md")
    docs_index = _read("docs/index.md")

    for content in (docs_readme, docs_index):
        assert "guides/snapback_workflow.md" in content
        assert "guides/snapback_released_workflow.md" in content
        assert "reference/snapback_artifacts.md" in content
        assert "reference/released_snapback_artifacts.md" in content
        assert "reference/release_enzyme_catalogs.md" in content
        assert "../workspaces/demo_snapback/README.md" in content
        assert "../workspaces/demo_snapback/runbook.md" in content


def test_runbook_step_reference_includes_released_snapback_demo_steps() -> None:
    runbook_steps = _read("docs/reference/runbook_steps.md")

    assert "`demo_snapback`" in runbook_steps
    assert "`snapback_released_design`" in runbook_steps
    assert "`snapback_released_show`" in runbook_steps
    assert "`snapback_released_target_search`" in runbook_steps
    assert "demo_released_origin_033.released.snapback.yaml" in runbook_steps
    assert "inputs/release_enzymes/local.release.yaml" in runbook_steps


def test_cli_and_reference_docs_capture_released_product_boundary() -> None:
    cli_ref = _read("docs/reference/cli.md")
    guide = _read("docs/guides/snapback_released_workflow.md")
    artifacts_ref = _read("docs/reference/released_snapback_artifacts.md")
    catalog_ref = _read("docs/reference/release_enzyme_catalogs.md")

    assert "cruncher snapback released-design" in cli_ref
    assert "cruncher snapback released-target-search" in cli_ref
    assert "cruncher snapback released-show" in cli_ref
    assert "type_iis_release_v1" in cli_ref

    assert "two-stage precursor" in guide
    assert "nick_then_release" in guide
    assert "retained post-release product" in guide
    assert "Type IIS enzymes are modeled here as release enzymes, not nickases." in guide
    assert "not a thermodynamic predictor and not a retron biology engine" in guide

    assert "outputs/released_design/" in artifacts_ref
    assert "released_snapback_manifest.json" in artifacts_ref
    assert "released_product_projection.json" in artifacts_ref
    assert "`released-show` is an integrity check" in artifacts_ref

    assert "BsaI-HFv2" in catalog_ref
    assert "BsmBI-v2" in catalog_ref
    assert "BbsI" in catalog_ref
    assert "SapI" in catalog_ref
