"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/tests/test_workspace_cli.py

Workspace list CLI contracts for cluster.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from dnadesign.cluster.src.cli.app import app

_RUNNER = CliRunner()


def test_workspaces_list_json_reports_builtin_workspace_state(monkeypatch, tmp_path: Path) -> None:
    workspaces_root = tmp_path / "workspaces"
    for workspace_id in ("perm_v1", "promoter_clusters_v1"):
        workspace_dir = workspaces_root / workspace_id
        workspace_dir.mkdir(parents=True, exist_ok=True)
        (workspace_dir / "config.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    (workspaces_root / "promoter_clusters_v1" / "outputs" / "cluster").mkdir(parents=True, exist_ok=True)
    (workspaces_root / "promoter_clusters_v1" / "outputs" / "cluster" / "run.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("dnadesign.cluster.src.workspaces.paths.builtin_workspaces_dir", lambda: workspaces_root)

    result = _RUNNER.invoke(app, ["workspaces", "list", "--format", "json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    by_id = {entry["workspace_id"]: entry for entry in payload}
    assert by_id["perm_v1"]["workspace_state"] == "clean"
    assert by_id["perm_v1"]["output_files"] == 0
    assert by_id["promoter_clusters_v1"]["workspace_state"] == "attention"
    assert by_id["promoter_clusters_v1"]["output_files"] == 1
    assert by_id["promoter_clusters_v1"]["latest_output_mtime"] is not None


def test_workspace_alias_list_json_reports_builtin_workspace_state(monkeypatch, tmp_path: Path) -> None:
    workspaces_root = tmp_path / "workspaces"
    for workspace_id in ("perm_v1", "promoter_clusters_v1"):
        workspace_dir = workspaces_root / workspace_id
        workspace_dir.mkdir(parents=True, exist_ok=True)
        (workspace_dir / "config.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    monkeypatch.setattr("dnadesign.cluster.src.workspaces.paths.builtin_workspaces_dir", lambda: workspaces_root)

    result = _RUNNER.invoke(app, ["workspace", "list", "--format", "json"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert [entry["workspace_id"] for entry in payload] == ["perm_v1", "promoter_clusters_v1"]
