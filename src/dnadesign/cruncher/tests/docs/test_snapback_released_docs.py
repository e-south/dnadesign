"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/tests/docs/test_snapback_released_docs.py

Docs contracts for the released-product snapback lane.

Module Author(s): Eric J. South
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
        assert "../workspaces/de033/README.md" in content
        assert "../workspaces/de033/runbook.md" in content


def test_runbook_step_reference_includes_released_snapback_de033_steps() -> None:
    runbook_steps = _read("docs/reference/runbook_steps.md")

    assert "`de033`" in runbook_steps
    assert "`snapback_released_solve`" in runbook_steps
    assert "`snapback_released_target_search`" in runbook_steps
    assert "thermo_nicking_v1" in runbook_steps
    assert "type_iis_release_v1" in runbook_steps
    assert "outputs/released_solve" in runbook_steps


def test_cli_and_reference_docs_capture_released_product_boundary() -> None:
    cli_ref = _read("docs/reference/cli.md")
    guide = _read("docs/guides/snapback_released_workflow.md")
    artifacts_ref = _read("docs/reference/released_snapback_artifacts.md")
    catalog_ref = _read("docs/reference/release_enzyme_catalogs.md")
    architecture_ref = _read("docs/reference/architecture.md")

    assert "cruncher snapback released-design" in cli_ref
    assert "cruncher snapback released-target-search" in cli_ref
    assert "cruncher snapback released-solve" in cli_ref
    assert "cruncher snapback released-show" in cli_ref
    assert "type_iis_release_v1" in cli_ref
    assert "thermo_nicking_v1" in cli_ref
    assert "requires at least one explicit nickase source and one explicit release-enzyme source" in cli_ref
    assert "--allow-demo-hits" in cli_ref
    assert "--allow-frequent-cutter-nickases" in cli_ref
    assert "--allow-top-active-routes" in cli_ref
    assert "--allow-precut-footprint-outside-active-product" in cli_ref
    assert "app/snapback_cli_requests.py" in cli_ref

    assert "two-stage precursor" in guide
    assert "nick_then_release" in guide
    assert "exposed post-release bottom strand" in guide
    assert "Type IIS enzymes are modeled here as release enzymes, not nickases." in guide
    assert "not a thermodynamic predictor and not a retron biology engine" in guide
    assert "Provide at least one explicit nickase source and one explicit release-enzyme source." in guide
    assert "--allow-top-active-routes" in guide
    assert "--allow-precut-footprint-outside-active-product" in guide
    assert "rebased origin" in guide
    assert "context left of the nick stays visible" in guide
    assert "demo-only" in guide.lower()
    assert "FREQUENT_CUTTER" in guide
    assert "neb_nicking_v1 + thermo_nicking_v1" in guide
    assert "physical top/bottom placement" in guide
    assert "retained bottom fragment" in guide
    assert "one representative per active-product `stem + cap` geometry" in guide
    assert "single contiguous fully degenerate `N`" in guide

    assert "outputs/released_design/" in artifacts_ref
    assert "outputs/released_solve/" in artifacts_ref
    assert "released_snapback_manifest.json" in artifacts_ref
    assert "released_product_projection.json" in artifacts_ref
    assert "released_solve_manifest.json" in artifacts_ref
    assert "`released-show` is an integrity check" in artifacts_ref
    assert "final-target drift" in artifacts_ref
    assert "physical top/bottom fragment-row placement" in artifacts_ref
    assert "retained_partner_strand" in artifacts_ref

    assert "BsaI-HFv2" in catalog_ref
    assert "BsmBI-v2" in catalog_ref
    assert "BbsI" in catalog_ref
    assert "SapI" in catalog_ref
    assert "BspQI" in catalog_ref

    assert "snapback/released_search/" in architecture_ref
    assert "snapback/released_target_search.py` is the thin public facade" in architecture_ref
    assert "snapback/released_explicit_evaluation.py" in architecture_ref
    assert "app/snapback_cli_requests.py" in architecture_ref
    assert "app/snapback_released_catalogs.py" in architecture_ref
    assert "app/snapback_released_show.py" in architecture_ref
    assert "app/snapback_released_show_{load,validate,present}.py" in architecture_ref
    assert "app/snapback_released_solve_workflow.py" in architecture_ref
    assert "app/snapback_released_solve_{snapshot,materialize,reporting}.py" in architecture_ref
    assert "cli/commands/snapback.py` is command registration only" in architecture_ref
