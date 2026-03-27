"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/docs/test_yiu_docs_contracts.py

Docs contracts for the YIU workflow family and family-aware routing.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_docs_index_routes_to_yiu_guide_and_references() -> None:
    docs_readme = _read("docs/README.md")
    docs_index = _read("docs/index.md")
    for content in (docs_readme, docs_index):
        assert "demos/demo_yiu_workspace.md" in content
        assert "guides/yiu_workflow.md" in content
        assert "reference/yiu_spec.md" in content
        assert "reference/yiu_artifacts.md" in content


def test_top_level_docs_route_three_workflow_families() -> None:
    package_readme = _read("README.md")
    docs_readme = _read("docs/README.md")
    docs_index = _read("docs/index.md")

    assert "YIU hairpin oligo" in package_readme
    assert "docs/demos/demo_yiu_workspace.md" in package_readme
    assert "docs/guides/yiu_workflow.md" in package_readme

    for content in (docs_readme, docs_index):
        assert "Model YIU Hairpin Oligo Processing" in content
        assert "demos/demo_yiu_workspace.md" in content
        assert "guides/yiu_workflow.md" in content


def test_cli_reference_lists_yiu_commands_and_contracts() -> None:
    cli_ref = _read("docs/reference/cli.md")

    assert "YIU workflows" in cli_ref
    assert "cruncher yiu validate" in cli_ref
    assert "cruncher yiu design" in cli_ref
    assert "cruncher yiu trace" in cli_ref
    assert "cruncher yiu init-workspace" in cli_ref
    assert "cruncher yiu show" in cli_ref
    assert ".yiu.yaml" in cli_ref


def test_yiu_docs_capture_workspace_and_artifact_boundaries() -> None:
    demo = _read("docs/demos/demo_yiu_workspace.md")
    guide = _read("docs/guides/yiu_workflow.md")
    spec_ref = _read("docs/reference/yiu_spec.md")
    artifacts_ref = _read("docs/reference/yiu_artifacts.md")
    architecture = _read("docs/reference/architecture.md")
    glossary = _read("docs/reference/glossary.md")

    assert "uv run cruncher yiu init-workspace" in demo
    assert "configs/yiu/example.yiu.yaml" in demo
    assert "outputs/yiu/explicit" in demo
    assert "published/views" in demo
    assert "state graph" in guide.lower()
    assert "source_oligo_ssdna" in guide
    assert "downstream_amplifiable_product" in guide
    assert "assembled_payload" in spec_ref
    assert "protocol_template" in spec_ref
    assert "msd_hop_retron_eco1_v1" in spec_ref
    assert "publish_contract_version" in spec_ref
    assert "min_paired_nt" in spec_ref
    assert "max_unpaired_tail_nt" in spec_ref
    assert "max_bulge_nt" in spec_ref
    assert "payload_junction_segments" in spec_ref
    assert "pattern_compatibility" in spec_ref
    assert "pattern_evidence_summary" in spec_ref
    assert "RETAINED_SACRIFICIAL_OVERLAP" in spec_ref
    assert "HOMOLOGY_WINDOW_SPANS_JUNCTION" in spec_ref
    assert "step_graph" in spec_ref
    assert "hairpin_pcr_linear_insert" in guide
    assert "protocol_template" in artifacts_ref
    assert "view_contract_version" in artifacts_ref
    assert "yiu_report.json" in artifacts_ref
    assert "yiu_trace.jsonl" in artifacts_ref
    assert "yiu_trace_manifest.json" in artifacts_ref
    assert "yiu_published_views_manifest.json" in artifacts_ref
    assert "validation_mode" in artifacts_ref
    assert "branch_junction" in artifacts_ref
    assert "bulge_nt" in artifacts_ref
    assert "spans_junction" in artifacts_ref
    assert "topology_compatibility" in artifacts_ref
    assert "sequence_mode" in guide
    assert "partial_complement" in guide
    assert "bulged" in guide
    assert "parts[]" in guide
    assert "`yiu/` (protocol-state YIU domain)" in architecture
    assert "retained product" in glossary.lower()
    assert "workflow family" in glossary.lower()
