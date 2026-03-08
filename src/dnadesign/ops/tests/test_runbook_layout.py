"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/tests/test_runbook_layout.py

Contract tests for workspace layout enforcement in ops runbook loading.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from dnadesign.ops.runbooks import runbook_layout


def test_enforce_workspace_layout_rejects_stdout_dir_outside_ops_logs(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    runbook = SimpleNamespace(
        id="demo_run",
        workspace_root=workspace_root,
        logging=SimpleNamespace(stdout_dir=(tmp_path / "outside" / "logs")),
        densegen=None,
        infer=None,
        notify=None,
    )

    with pytest.raises(ValueError, match="logging.stdout_dir must be under"):
        runbook_layout.enforce_workspace_layout(runbook)
