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
        assert "yiu init-workspace|validate|design|trace|solve|show" in content


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
    assert "cruncher yiu solve" in cli_ref
    assert "cruncher yiu init-workspace" in cli_ref
    assert "cruncher yiu show" in cli_ref
    assert ".yiu.yaml" in cli_ref
    assert ".yiu.solve.yaml" in cli_ref


def test_yiu_docs_capture_workspace_and_artifact_boundaries() -> None:
    demo = _read("docs/demos/demo_yiu_workspace.md")
    guide = _read("docs/guides/yiu_workflow.md")
    spec_ref = _read("docs/reference/yiu_spec.md")
    artifacts_ref = _read("docs/reference/yiu_artifacts.md")
    architecture = _read("docs/reference/architecture.md")
    glossary = _read("docs/reference/glossary.md")

    assert "src/dnadesign/cruncher/workspaces/demo_yiu_circularized" in demo
    assert "uv run cruncher workspaces run --workspace demo_yiu_circularized --runbook configs/runbook.yaml" in demo
    assert "uv run cruncher yiu init-workspace yiu_lab_demo" in demo
    assert "configs/yiu/example_canonical_circularized.yiu.yaml" in demo
    assert "configs/yiu/example_canonical_circularized.yiu.solve.yaml" in demo
    assert "outputs/yiu/explicit" in demo
    assert "outputs/yiu/solve" in demo
    assert "published/views" in demo
    assert "published/baserender_jobs" in demo
    assert "uv run cruncher visuals validate" in demo
    assert "uv run cruncher visuals run" in demo
    assert "state graph" in guide.lower()
    assert "source_oligo_ssdna" in guide
    assert "hairpin_pcr_linear_insert" in guide
    assert "assembled_payload" in spec_ref
    assert "protocol_template" in spec_ref
    assert "yiu_adapter_hairpin_v1" in spec_ref
    assert "yiu_circularized_payload_v1" in spec_ref
    assert "yiu_solve" in spec_ref
    assert "publish_contract_version" in spec_ref
    assert "emit_baserender_jobs" in spec_ref
    assert "circularized_payload_junction" in spec_ref
    assert "accepted_hits.jsonl" in artifacts_ref
    assert "yiu_solve_report.json" in artifacts_ref
    assert "published/visual_manifest.json" in artifacts_ref
    assert "materialized hit" in artifacts_ref.lower()
    assert "design" in guide
    assert "trace" in guide
    assert "solve" in guide
    assert "protocol_template" in artifacts_ref
    assert "yiu_report.json" in artifacts_ref
    assert "yiu_trace.jsonl" in artifacts_ref
    assert "published/views/" in artifacts_ref
    assert "published/renders/" in artifacts_ref
    assert "bundle kind" in guide.lower()
    assert "`yiu/` (protocol-state YIU domain)" in architecture
    assert "yiu init-workspace|validate|design|trace|solve|show" in architecture
    assert "**yiu solve**" in architecture
    assert "retained product" in glossary.lower()
    assert "workflow family" in glossary.lower()


def test_checked_in_canonical_yiu_demo_workspace_exists() -> None:
    workspace_root = ROOT / "workspaces" / "demo_yiu_circularized"

    assert workspace_root.exists()
    assert (workspace_root / "runbook.md").exists()
    assert (workspace_root / "configs" / "runbook.yaml").exists()
    assert (workspace_root / "configs" / "yiu" / "example_canonical_circularized.yiu.yaml").exists()
    assert (workspace_root / "configs" / "yiu" / "example_canonical_circularized.yiu.solve.yaml").exists()
    assert (workspace_root / "catalogs" / "enzymes.yaml").exists()
    assert (workspace_root / "catalogs" / "oligo_parts.yaml").exists()
    assert (workspace_root / "catalogs" / "backbones.yaml").exists()
    runbook_doc = (workspace_root / "runbook.md").read_text(encoding="utf-8")
    assert "Canonical checked-in YIU demo" in runbook_doc
    assert (
        "uv run cruncher workspaces run --workspace demo_yiu_circularized --runbook configs/runbook.yaml" in runbook_doc
    )
    assert "uv run cruncher visuals run" in runbook_doc
