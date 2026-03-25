"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/docs/test_cassette_docs_contracts.py

Docs contracts for the cassette workflow and reference routing.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_docs_index_routes_to_cassette_guide_and_references() -> None:
    docs_readme = _read("docs/README.md")
    docs_index = _read("docs/index.md")
    for content in (docs_readme, docs_index):
        assert "guides/cassette_workflow.md" in content
        assert "guides/cassette_solve_workflow.md" in content
        assert "reference/cassette_spec.md" in content
        assert "reference/cassette_solve_spec.md" in content
        assert "reference/nickase_catalog.md" in content
        assert "reference/cassette_artifacts.md" in content


def test_cli_reference_lists_cassette_commands_and_contracts() -> None:
    cli_ref = _read("docs/reference/cli.md")
    assert "Cassette workflows" in cli_ref
    assert "cruncher cassette validate" in cli_ref
    assert "cruncher cassette design" in cli_ref
    assert "cruncher cassette solve" in cli_ref
    assert "cruncher cassette catalog init-neb" in cli_ref
    assert "cruncher cassette show" in cli_ref
    assert ".cassette.yaml" in cli_ref
    assert ".cassette.solve.yaml" in cli_ref
    assert "no fallback to `sample`" in cli_ref


def test_cassette_guide_states_current_scope_and_outputs() -> None:
    guide = _read("docs/guides/cassette_workflow.md")
    solve_guide = _read("docs/guides/cassette_solve_workflow.md")
    assert "explicit lane does not search over stems, loops, or nickase assignments" in guide
    assert "outputs/cassettes/<spec.name>/<design_id>/" in guide
    assert "bounded_nicked_segment" in guide
    assert "render_contract.json" in guide
    assert "RIGHT_WINDOW_NO_MATCH" in guide
    assert "outputs/cassette_solves/<solve_id>/" in solve_guide
    assert "max_search_nodes" in solve_guide
    assert "per-hit explicit cassette bundles" in solve_guide


def test_cassette_references_capture_schema_and_artifacts() -> None:
    spec_ref = _read("docs/reference/cassette_spec.md")
    solve_spec_ref = _read("docs/reference/cassette_solve_spec.md")
    catalog_ref = _read("docs/reference/nickase_catalog.md")
    artifacts_ref = _read("docs/reference/cassette_artifacts.md")
    assert "nick_window.start" in spec_ref
    assert "boundary_inclusive_v2" in spec_ref
    assert "derived_reverse_complement" in spec_ref
    assert "max_search_nodes" in solve_spec_ref
    assert "forbidden_any_site_specificity_ids" in solve_spec_ref
    assert "neb_nicking_v1" in solve_spec_ref
    assert "motif_top_5to3" in catalog_ref
    assert "product_aliases" in catalog_ref
    assert "WarmStart Nt.BstNBI" in catalog_ref
    assert "raw_cut_notation" in catalog_ref
    assert "top_cut_offset" in catalog_ref
    assert "cassette catalog init-neb" in catalog_ref
    assert "cassette_manifest.json" in artifacts_ref
    assert "solve_report.json" in artifacts_ref
    assert "table__hits.csv" in artifacts_ref
    assert "do not register in workspace `run_index.json`" in artifacts_ref


def test_architecture_and_glossary_capture_cassette_boundary() -> None:
    architecture = _read("docs/reference/architecture.md")
    glossary = _read("docs/reference/glossary.md")
    assert "#### Cassette lifecycle" in architecture
    assert "#### `cassette/` (dual-context cassette domain)" in architecture
    assert "render_contract.json" in architecture
    assert "bounded nicked segment" in glossary.lower()
    assert "pair map" in glossary.lower()
    assert "target strand" in glossary.lower()
