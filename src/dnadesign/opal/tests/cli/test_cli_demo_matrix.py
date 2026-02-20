"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/cli/test_cli_demo_matrix.py

CLI tests for demo workflow matrix command wiring and summary output.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.src.cli.commands import demo_matrix as demo_matrix_cmd


def test_demo_matrix_json_summary_shape(monkeypatch, tmp_path: Path) -> None:
    def _fake_run_demo_flow(*, flow_name: str, tmp_root: Path, rounds: list[int], fail_fast: bool) -> dict:
        _ = tmp_root
        _ = fail_fast
        return {
            "flow": flow_name,
            "ok": True,
            "rounds": [{"round": r, "mismatch_count": 0} for r in rounds],
        }

    monkeypatch.setattr(demo_matrix_cmd, "_run_demo_flow", _fake_run_demo_flow)
    app = _build()
    runner = CliRunner()
    res = runner.invoke(
        app,
        ["--no-color", "demo-matrix", "--tmp-root", str(tmp_path), "--rounds", "0,1", "--json"],
    )
    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)
    assert out["ok"] is True
    assert out["rounds"] == [0, 1]
    assert len(out["flows"]) == len(demo_matrix_cmd.DEMO_FLOWS)
