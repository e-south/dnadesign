"""
--------------------------------------------------------------------------------
dnadesign
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
        assert "demos/demo_cassette_workspace.md" in content
        assert "guides/cassette_workflow.md" in content
        assert "guides/cassette_solve_workflow.md" in content
        assert "reference/cassette_spec.md" in content
        assert "reference/cassette_solve_spec.md" in content
        assert "reference/nickase_catalog.md" in content
        assert "reference/cassette_artifacts.md" in content


def test_top_level_docs_route_both_workflow_families() -> None:
    package_readme = _read("README.md")
    docs_readme = _read("docs/README.md")
    docs_index = _read("docs/index.md")

    assert "Cruncher runs reproducible DNA design jobs" in package_readme
    assert "DNA design package in `dnadesign`" not in package_readme
    assert "cassette design" in package_readme
    assert "inspect payload windows" in package_readme
    assert "docs/README.md" in package_readme
    assert "docs/demos/demo_cassette_workspace.md" in package_readme
    assert "docs/guides/sampling_and_analysis.md" in package_readme
    assert "docs/guides/yiu_workflow.md" in package_readme
    assert "docs/reference/cli.md" in package_readme
    assert "This README stays light on purpose." not in package_readme
    assert "Aggregate parameter sweeps" in package_readme

    assert "Optimize Fixed-Length Sequences" in docs_readme
    assert "Design and Search Cassettes" in docs_readme
    assert "demos/demo_pairwise.md" in docs_readme
    assert "demos/demo_multitf.md" in docs_readme
    assert "demos/project_all_tfs.md" in docs_readme
    assert "demos/demo_cassette_workspace.md" in docs_readme
    assert "guides/sampling_and_analysis.md" in docs_readme
    assert "guides/studies.md" in docs_readme
    assert "guides/portfolio_aggregation.md" in docs_readme

    assert "Design and Search Cassettes" in docs_index
    assert "Optimize Fixed-Length Sequences" in docs_index
    assert "Summarize Sweeps and Aggregate Artifacts" in docs_index
    assert "demos/demo_pairwise.md" in docs_index
    assert "demos/demo_multitf.md" in docs_index
    assert "demos/project_all_tfs.md" in docs_index
    assert "demos/demo_cassette_workspace.md" in docs_index
    assert "guides/sampling_and_analysis.md" in docs_index
    assert "guides/studies.md" in docs_index
    assert "guides/portfolio_aggregation.md" in docs_index


def test_cassette_demo_defines_scaffolded_workspace_flow() -> None:
    demo = _read("docs/demos/demo_cassette_workspace.md")
    assert "WORKSPACES_ROOT=src/dnadesign/cruncher/workspaces" in demo  # pragma: allowlist secret
    assert "DEMO_WORKSPACE=cassette_lab_demo" in demo
    assert 'uv run cruncher cassette init-workspace "$DEMO_WORKSPACE"' in demo
    assert 'uv run cruncher workspaces list --root "$WORKSPACES_ROOT"' in demo
    assert "demo_hairpin_fast.cassette.solve.yaml" in demo
    assert "cassette_workspace_manifest.json" in demo
    assert "configs/runbook.yaml" in demo
    assert "runbook-only" in demo
    assert "views/" in demo
    assert "baserender_jobs/" in demo
    assert "renders/" in demo
    assert "uv run baserender job validate" in demo
    assert "uv run baserender job run" in demo
    assert "../guides/cassette_solve_workflow.md" in demo
    assert "../reference/cassette_artifacts.md" in demo


def test_cli_reference_lists_cassette_commands_and_contracts() -> None:
    cli_ref = _read("docs/reference/cli.md")
    assert "Cassette workflows" in cli_ref
    assert "cruncher cassette validate" in cli_ref
    assert "cruncher cassette design" in cli_ref
    assert "cruncher cassette solve" in cli_ref
    assert "cruncher cassette init-workspace" in cli_ref
    assert "cruncher cassette catalog init-neb" in cli_ref
    assert "cruncher cassette show" in cli_ref
    assert ".cassette.yaml" in cli_ref
    assert ".cassette.solve.yaml" in cli_ref
    assert "no fallback to `sample`" in cli_ref
    assert "score_only" in cli_ref
    assert "greedy_hamming" in cli_ref
    assert "ACCEPTED_POOL_TRUNCATED" in cli_ref
    assert "SELECTION_POLICY_LIMITED_HITS" in cli_ref


def test_cassette_guide_states_current_scope_and_outputs() -> None:
    guide = _read("docs/guides/cassette_workflow.md")
    solve_guide = _read("docs/guides/cassette_solve_workflow.md")
    assert "explicit lane does not search over stems, loops, or nickase assignments" in guide
    assert "../demos/demo_cassette_workspace.md" in guide
    assert "outputs/cassettes/<spec.name>/<design_id>/" in guide
    assert "bounded_nicked_segment" in guide
    assert "views/linear_duplex.v1.json" in guide
    assert "baserender_jobs/linear_duplex.job.yaml" in guide
    assert "renders/linear_duplex.pdf" in guide
    assert "RIGHT_WINDOW_NO_MATCH" in guide
    assert "outputs/cassette_solves/<solve_id>/" in solve_guide
    assert "init-workspace" in solve_guide
    assert "demo_hairpin_fast.cassette.solve.yaml" in solve_guide
    assert "configs/runbook.yaml" in solve_guide
    assert "runbook-only" in solve_guide
    assert "max_search_nodes" in solve_guide
    assert "per-hit explicit cassette bundles" in solve_guide
    assert "score_only" in solve_guide
    assert "greedy_hamming" in solve_guide
    assert "mmr" in solve_guide
    assert "accepted pool" in solve_guide.lower()
    assert "views/top_hits.linear_duplex.v1.jsonl" in solve_guide
    assert "baserender_jobs/top_hits_duplex.job.yaml" in solve_guide
    assert "../demos/demo_cassette_workspace.md" in solve_guide
    assert (
        "uv run baserender job run "
        "outputs/cassette_solves/<solve_id>/baserender_jobs/top_hits_duplex.job.yaml" in solve_guide
    )
    assert "hits/hit_<rank>_<solution_id>/renders/ssdna_hairpin.pdf" in solve_guide


def test_cassette_references_capture_schema_and_artifacts() -> None:
    spec_ref = _read("docs/reference/cassette_spec.md")
    solve_spec_ref = _read("docs/reference/cassette_solve_spec.md")
    catalog_ref = _read("docs/reference/nickase_catalog.md")
    artifacts_ref = _read("docs/reference/cassette_artifacts.md")
    assert "nick_window.start" in spec_ref
    assert "boundary_inclusive_v2" in spec_ref
    assert "derived_reverse_complement" in spec_ref
    assert "max_search_nodes" in solve_spec_ref
    assert "search.selection" in solve_spec_ref
    assert "diversity_weight" in solve_spec_ref
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
    assert "views/views_manifest.v1.json" in artifacts_ref
    assert "top_hits.linear_duplex.v1.jsonl" in artifacts_ref
    assert "top_hits_duplex_qa_sheet.pdf" in artifacts_ref
    assert "The cassette render path is intentionally local to the owning workspace" in artifacts_ref
    assert "per-hit `renders/*.pdf`" in artifacts_ref
    assert "selection_summary" in artifacts_ref
    assert "ACCEPTED_POOL_TRUNCATED" in artifacts_ref
    assert "SELECTION_POLICY_LIMITED_HITS" in artifacts_ref
    assert "do not register in workspace `run_index.json`" in artifacts_ref


def test_architecture_and_glossary_capture_cassette_boundary() -> None:
    architecture = _read("docs/reference/architecture.md")
    glossary = _read("docs/reference/glossary.md")
    assert "#### Cassette lifecycle" in architecture
    assert "#### `cassette/` (dual-context cassette domain)" in architecture
    assert "views/linear_duplex.v1.json" in architecture
    assert "bounded nicked segment" in glossary.lower()
    assert "pair map" in glossary.lower()
    assert "target strand" in glossary.lower()
