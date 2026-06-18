"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/tests/docs/test_yiu_demo_workspace_contracts.py

Checked-in demo workspace contracts for payload-centric YIU docs surfaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_demo_doc_points_to_checked_in_workspace() -> None:
    demo = _read("docs/demos/demo_yiu_workspace.md")

    assert "src/dnadesign/cruncher/workspaces/demo_yiu_payload" in demo
    assert "uv run cruncher workspaces run --workspace demo_yiu_payload --runbook configs/runbook.yaml" in demo
    assert 'uv run cruncher yiu render --spec "$USER_SPEC" --force-overwrite --emit-renders' in demo
    assert 'uv run cruncher yiu show --bundle "$DEMO_WORKSPACE/outputs/example_payload"' in demo
    assert "Read [YIU Workflow](../guides/yiu_workflow.md) next" in demo
    assert "[YIU Artifacts](../reference/yiu_artifacts.md)" in demo
    assert "[YIU Visual System](../reference/yiu_visual_system.md)" in demo


def test_checked_in_yiu_demo_workspace_exists() -> None:
    workspace_root = ROOT / "workspaces" / "demo_yiu_payload"

    assert workspace_root.exists()
    assert (workspace_root / "runbook.md").exists()
    assert (workspace_root / "configs" / "runbook.yaml").exists()
    assert (workspace_root / "configs" / "yiu" / "example_payload.yiu.yaml").exists()
    assert (workspace_root / "configs" / "yiu" / "example_payload.advanced_pwm.example.yaml").exists()
    assert (workspace_root / "motifs" / "example_pwm_context.yaml").exists()
    runbook_doc = (workspace_root / "runbook.md").read_text(encoding="utf-8")
    assert "Checked-in YIU demo for the v4 payload optimization and rendering workflow." in runbook_doc
    assert "uv run cruncher yiu render" in runbook_doc
    assert "uv run cruncher yiu show" in runbook_doc
