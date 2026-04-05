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
        assert "reference/yiu_visual_system.md" in content
        assert "Ownership split:" in content
        assert "yiu init-workspace|validate|render|show" in content
        assert "trace|solve" not in content


def test_top_level_docs_route_three_workflow_families() -> None:
    package_readme = _read("README.md")
    docs_readme = _read("docs/README.md")
    docs_index = _read("docs/index.md")

    assert "payload-centric YIU" in package_readme
    assert "outputs/<workflow>/" in package_readme
    assert "bundles/<workflow>/" not in package_readme
    assert "docs/demos/demo_yiu_workspace.md" in package_readme
    assert "docs/guides/yiu_workflow.md" in package_readme

    for content in (docs_readme, docs_index):
        assert "Payload-Centric YIU Workflows" in content
        assert "demos/demo_yiu_workspace.md" in content
        assert "guides/yiu_workflow.md" in content
        assert "reference/yiu_visual_system.md" in content


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


def test_yiu_docs_route_readers_to_their_primary_reference_pages() -> None:
    demo = _read("docs/demos/demo_yiu_workspace.md")
    guide = _read("docs/guides/yiu_workflow.md")
    spec_ref = _read("docs/reference/yiu_spec.md")
    artifacts_ref = _read("docs/reference/yiu_artifacts.md")
    visual_ref = _read("docs/reference/yiu_visual_system.md")
    architecture = _read("docs/reference/architecture.md")
    runbook_steps = _read("docs/reference/runbook_steps.md")
    workspaces_readme = _read("workspaces/README.md")

    assert "src/dnadesign/cruncher/workspaces/demo_yiu_payload" in demo
    assert "uv run cruncher workspaces run --workspace demo_yiu_payload --runbook configs/runbook.yaml" in demo
    assert "uv run cruncher yiu init-workspace yiu_lab_demo" in demo
    assert ".yiu.yaml" in demo
    assert ".yiu.solve.yaml" not in demo
    assert "outputs/example_payload/" in demo
    assert "uv run cruncher yiu render" in demo
    assert "demo_monotypic_tetr" in demo
    assert "[YIU Artifacts](../reference/yiu_artifacts.md)" in demo
    assert "[YIU Visual System](../reference/yiu_visual_system.md)" in demo
    assert "Use this guide for command flow and operator posture." in guide
    assert "### Documentation ownership" in guide
    assert "[YIU Spec Reference](../reference/yiu_spec.md)" in guide
    assert "[YIU Artifacts](../reference/yiu_artifacts.md)" in guide
    assert "[YIU Visual System](../reference/yiu_visual_system.md)" in guide
    assert "[Cruncher architecture](../reference/architecture.md)" in guide
    assert "strict v4 contract" in guide
    assert "### Start here" in guide
    assert "### Bundle surface" in guide
    assert "optimized junction/mismatch plan" in guide
    assert "output.bundle_dir" in guide
    assert "outputs/<workflow>/" in guide
    assert "shared bundle-artifact surface and one shared bundle-state family" in guide
    assert "The payload view uses `yiu_payload_visual_v1`." in guide
    assert "The current YIU visual system is `bench_strip`" in guide
    assert "motif layers aligned to payload-forward coordinates" in guide
    assert (
        "bundle contract is intentionally split across bundle truth, published "
        "view contracts, and composite render output"
    ) in guide
    assert ("exact emitted files, shared inspection fields, and render-status semantics") in guide
    assert "### Maintainer boundaries" in guide
    assert "[Architecture](../reference/architecture.md)" in guide
    assert "[YIU Artifacts](../reference/yiu_artifacts.md)" in guide
    assert "Cross-tool integrations should not import" in guide
    assert "baserender.src.*" in guide
    assert "Ambiguous or missing sources fail fast." in guide
    assert "split_yiu_payload_rendering_v4" in spec_ref
    assert "This page owns schema and normalization only." in spec_ref
    assert "operator flow and visual posture" in spec_ref
    assert "Bundle layout, render-status semantics, and operator inspection fields live in" in spec_ref
    assert "current workspace root or its parent directory" in spec_ref
    assert "split_yiu_payload_bundle_v4" in artifacts_ref
    assert "source of truth for emitted files" in artifacts_ref
    assert "shared `render`/`show` inspection surface" in artifacts_ref
    assert "[YIU Visual System](yiu_visual_system.md)" in artifacts_ref
    assert "visual_inventory.json" in artifacts_ref
    assert "visual_direction" in artifacts_ref
    assert "render_status: rendered" in artifacts_ref
    assert "render_status: failed" in artifacts_ref
    assert "bundle_dir" in artifacts_ref
    assert "outputs_root" in artifacts_ref
    assert "bundle_manifest_path" in artifacts_ref
    assert "normalized_payload_path" in artifacts_ref
    assert "visual_inventory_path" in artifacts_ref
    assert "Published contract paths" in artifacts_ref
    assert "Use [YIU Workflow](../guides/yiu_workflow.md)" in artifacts_ref
    assert "### Shared bundle surface" in artifacts_ref
    assert "cruncher yiu render" in artifacts_ref
    assert "cruncher yiu show" in artifacts_ref
    assert "shared `render`/`show` inspection surface" in artifacts_ref
    assert "render-status semantics" in artifacts_ref
    assert "bundle layout changes should land once in the app layer" in artifacts_ref
    assert ("shared manifest/inventory/normalized load-persist helpers live in `yiu/bundle_state.py`") in architecture
    assert (
        "shared typed render/show bundle-artifact surfaces for app/CLI boundaries live in `yiu/bundle_surface.py`"
    ) in architecture
    assert "published-contract BaseRender execution lives in `yiu/render.py`" in architecture
    assert "transactional render-plan execution" in architecture
    assert "app/yiu_workflow/render.py" in architecture
    assert "app/yiu_workflow/show.py" in architecture
    assert "payload bundle publication orchestration lives in `yiu/publish.py`" in architecture
    assert "canonical view-entry/render-job planning lives in `yiu/view_catalog.py`" in architecture
    assert "display/title policy plus named visual directions live in `yiu/view_styles.py`" in architecture
    assert "the named YIU visual system is `bench_strip`" in architecture
    assert "This page owns the named visual directions and information hierarchy" in visual_ref
    assert "`bench_strip`" in visual_ref
    assert "`payload` uses `evidence_ribbon`" in visual_ref
    assert "`split_payload` uses `operator_strip`" in visual_ref
    assert "`assembled_payload` uses `operator_strip`" in visual_ref
    assert "producer-side style policy lives in `src/dnadesign/cruncher/src/yiu/visual_system.py`" in visual_ref
    assert "shared view fragments live in `yiu/view_common.py`" in architecture
    assert "`yiu/` (payload-centric YIU domain)" in architecture
    assert "cruncher yiu render --spec configs/yiu/example_payload.yiu.yaml" in runbook_steps
    assert "cruncher yiu show --bundle outputs/example_payload" in runbook_steps
    assert "outputs/\n      <workflow>/" in workspaces_readme


def test_yiu_docs_keep_render_state_and_artifact_surface_contracts_separate() -> None:
    guide = _read("docs/guides/yiu_workflow.md")
    artifacts_ref = _read("docs/reference/yiu_artifacts.md")
    spec_ref = _read("docs/reference/yiu_spec.md")

    assert (
        "Use [YIU Artifacts](../reference/yiu_artifacts.md) for the exact "
        "emitted files, shared inspection fields, and render-status semantics."
    ) in guide
    assert "[YIU Visual System](../reference/yiu_visual_system.md)" in guide
    assert (
        "bundle contract is intentionally split across bundle truth, "
        "published view contracts, and composite render output"
    ) in guide
    assert "This page owns schema and normalization only." in spec_ref
    assert "Bundle layout, render-status semantics, and operator inspection fields live in" in spec_ref
    assert "operator flow and visual posture" in spec_ref
    assert "source of truth for emitted files" in artifacts_ref
    assert "Each bundle uses `visual_inventory.json` as the operator-facing render-state record." in artifacts_ref
    assert "visual_direction" in artifacts_ref
    assert "YIU Visual System" in artifacts_ref
    assert "cruncher yiu render" in artifacts_ref
    assert "cruncher yiu show" in artifacts_ref
    assert "render-status semantics" in artifacts_ref
    assert "published artifact paths disagree" in artifacts_ref


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
