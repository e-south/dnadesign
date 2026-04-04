"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/docs/test_yiu_docs_contracts.py

Docs contracts for the payload-centric YIU workflow family.

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
        assert "yiu init-workspace|validate|render|show" in content
        assert "trace|solve" not in content


def test_top_level_docs_route_three_workflow_families() -> None:
    package_readme = _read("README.md")
    docs_readme = _read("docs/README.md")
    docs_index = _read("docs/index.md")

    assert "payload-centric YIU" in package_readme
    assert "docs/demos/demo_yiu_workspace.md" in package_readme
    assert "docs/guides/yiu_workflow.md" in package_readme

    for content in (docs_readme, docs_index):
        assert "Render Split YIU Payloads" in content
        assert "demos/demo_yiu_workspace.md" in content
        assert "guides/yiu_workflow.md" in content


def test_cli_reference_lists_yiu_commands_and_contracts() -> None:
    cli_ref = _read("docs/reference/cli.md")

    assert "YIU workflows" in cli_ref
    assert "cruncher yiu validate" in cli_ref
    assert "cruncher yiu init-workspace" in cli_ref
    assert "cruncher yiu show" in cli_ref
    assert "cruncher yiu render" in cli_ref
    assert ".yiu.yaml" in cli_ref
    assert ".yiu.solve.yaml" not in cli_ref
    assert "split_yiu_payload_rendering_v4" in cli_ref
    assert "<workspace>/outputs/<name>/" in cli_ref


def test_yiu_docs_capture_payload_workspace_and_artifact_boundaries() -> None:
    demo = _read("docs/demos/demo_yiu_workspace.md")
    guide = _read("docs/guides/yiu_workflow.md")
    spec_ref = _read("docs/reference/yiu_spec.md")
    artifacts_ref = _read("docs/reference/yiu_artifacts.md")
    architecture = _read("docs/reference/architecture.md")
    runbook_steps = _read("docs/reference/runbook_steps.md")
    workspaces_readme = _read("workspaces/README.md")

    assert "src/dnadesign/cruncher/workspaces/demo_yiu_payload" in demo
    assert "uv run cruncher workspaces run --workspace demo_yiu_payload --runbook configs/runbook.yaml" in demo
    assert "uv run cruncher yiu init-workspace yiu_lab_demo" in demo
    assert ".yiu.yaml" in demo
    assert ".yiu.solve.yaml" not in demo
    assert "outputs/example_payload/" in demo
    assert "normalized_payload.json" in demo
    assert "bundle_manifest.json" in demo
    assert "uv run cruncher yiu render" in demo
    assert "demo_monotypic_tetr" in demo
    assert "strict v4 contract" in guide
    assert "optimized junction/mismatch plan" in guide
    assert "selected payload strand, selected complement strand, optional PWM motif layers" in guide
    assert "The payload view uses `yiu_payload_visual_v1`." in guide
    assert "motif layers aligned to payload-forward coordinates" in guide
    assert "integrity checks against the manifest, inventory, normalized payload, and published view contracts" in guide
    assert "user_sequence" in guide
    assert "sample_hit" in guide
    assert "payload" in guide
    assert "split_payload" in guide
    assert "assembled_payload" in guide
    assert "4 nt junction window" in guide
    assert "point split" not in guide.lower()
    assert "right-then-left" not in guide.lower()
    assert "split_yiu_payload_rendering_v4" in spec_ref
    assert "sample_hit" in spec_ref
    assert "user_sequence" in spec_ref
    assert "bundle_dir" in spec_ref
    assert "published_plot_path" in spec_ref
    assert "candidate_positions" in spec_ref
    assert "default_strand_preference" in spec_ref
    assert "primary: maximin" in spec_ref
    assert "yiu_payload_visual_v1" in spec_ref
    assert "left_member" not in spec_ref
    assert "right_member" not in spec_ref
    assert "emit_view_contracts" not in spec_ref
    assert "split_yiu_payload_bundle_v4" in artifacts_ref
    assert "visual_inventory.json" in artifacts_ref
    assert "normalized_payload.json" in artifacts_ref
    assert "payload_view.json" in artifacts_ref
    assert "split_payload_view.json" in artifacts_ref
    assert "assembled_payload_view.json" in artifacts_ref
    assert "yiu_payload_visual_v1" in artifacts_ref
    assert "payload-forward coordinates" in artifacts_ref
    assert "ligation_junction" not in artifacts_ref
    assert "assembled seam" not in artifacts_ref.lower()
    assert "baserender_jobs/" in artifacts_ref
    assert "published_plot_artifact_path" in artifacts_ref
    assert "trace" not in guide
    assert "9-state" not in guide
    assert "protocol replay" not in guide.lower()
    assert "ship_v3" not in guide
    assert "ship_v4" not in guide
    assert "`yiu/` (payload-centric YIU domain)" in architecture
    assert "yiu init-workspace|validate|render|show" in architecture
    assert "trace|solve" not in architecture
    assert "split_yiu_payload_rendering_v4" in architecture
    assert "split_yiu_payload_bundle_v4" in architecture
    assert (
        "cruncher yiu render --spec configs/yiu/example_payload.yiu.yaml --force-overwrite --emit-renders"
        in runbook_steps
    )
    assert "cruncher yiu show --bundle outputs/example_payload" in runbook_steps
    assert (
        "cruncher yiu render --spec configs/yiu/tetr_monotypic_hit.yiu.yaml --force-overwrite --emit-renders"
        in runbook_steps
    )
    assert "cruncher yiu show --bundle outputs/yiu__tetr_monotypic_hit" in runbook_steps
    runbook_yaml = _read("workspaces/demo_yiu_payload/configs/runbook.yaml")
    assert "description: Validate the checked-in user-sequence YIU demo spec." in runbook_yaml
    assert (
        "description: Publish the deterministic user-sequence YIU v4 payload bundle and render the canonical views."
        in runbook_yaml
    )
    assert "description: Inspect the published user-sequence payload bundle and integrity checks." in runbook_yaml
    assert "<workflow>.yiu.solve.yaml" not in workspaces_readme
    assert "outputs/\n      <workflow>/" in workspaces_readme


def test_checked_in_yiu_demo_workspace_exists() -> None:
    workspace_root = ROOT / "workspaces" / "demo_yiu_payload"

    assert workspace_root.exists()
    assert (workspace_root / "runbook.md").exists()
    assert (workspace_root / "configs" / "runbook.yaml").exists()
    assert (workspace_root / "configs" / "yiu" / "example_payload.yiu.yaml").exists()
    assert (workspace_root / "motifs" / "example_pwm_context.yaml").exists()
    assert not (workspace_root / "configs" / "yiu" / "example_payload.yiu.solve.yaml").exists()
    assert not (workspace_root / "catalogs").exists()
    runbook_doc = (workspace_root / "runbook.md").read_text(encoding="utf-8")
    assert "Checked-in YIU demo for the v4 payload optimization and rendering workflow." in runbook_doc
    assert "user-sequence-only" in runbook_doc
    assert "uv run cruncher workspaces run --workspace demo_yiu_payload --runbook configs/runbook.yaml" in runbook_doc
    assert "uv run cruncher yiu render" in runbook_doc
    assert "uv run cruncher yiu show" in runbook_doc
