"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_paths.py

Contract tests for CLI path rendering behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.cli.paths import render_path


def test_render_path_returns_relative_for_in_workspace_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    target = workspace / "outputs" / "report.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("ok", encoding="utf-8")

    rendered = render_path(target, base=workspace)
    assert rendered == "outputs/report.md"


def test_render_path_returns_absolute_for_far_outside_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace" / "deep" / "nest"
    workspace.mkdir(parents=True, exist_ok=True)
    external = tmp_path / "external" / "outputs" / "result.json"
    external.parent.mkdir(parents=True, exist_ok=True)
    external.write_text("{}", encoding="utf-8")

    rendered = render_path(external, base=workspace)
    assert rendered == str(external.resolve())
