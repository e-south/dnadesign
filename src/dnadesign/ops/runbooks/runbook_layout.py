"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/runbooks/runbook_layout.py

Workspace layout enforcement for resolved ops runbook paths.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import OrchestrationRunbookV1


def _is_path_within(*, path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def enforce_workspace_layout(runbook: "OrchestrationRunbookV1") -> "OrchestrationRunbookV1":
    workspace_root = runbook.workspace_root.resolve()
    ops_logs_root = (workspace_root / "outputs" / "logs" / "ops" / "sge").resolve()
    if not _is_path_within(path=runbook.logging.stdout_dir, parent=ops_logs_root):
        raise ValueError(f"logging.stdout_dir must be under {ops_logs_root}")
    expected_stdout_dir = (ops_logs_root / runbook.id).resolve()
    if runbook.logging.stdout_dir.resolve() != expected_stdout_dir:
        raise ValueError(f"logging.stdout_dir must be exactly {expected_stdout_dir}")

    if runbook.densegen is not None:
        expected_config = (workspace_root / "config.yaml").resolve()
        if runbook.densegen.config.resolve() != expected_config:
            raise ValueError(f"densegen.config must be {expected_config}")
    if runbook.infer is not None:
        expected_config = (workspace_root / "config.yaml").resolve()
        if runbook.infer.config.resolve() != expected_config:
            raise ValueError(f"infer.config must be {expected_config}")
    if runbook.notify is not None:
        notify_root = (workspace_root / "outputs" / "notify" / runbook.notify.tool).resolve()
        expected_profile = (notify_root / "profile.json").resolve()
        expected_cursor = (notify_root / "cursor").resolve()
        expected_spool = (notify_root / "spool").resolve()
        if runbook.notify.profile.resolve() != expected_profile:
            raise ValueError(f"notify.profile must be {expected_profile}")
        if runbook.notify.cursor.resolve() != expected_cursor:
            raise ValueError(f"notify.cursor must be {expected_cursor}")
        if runbook.notify.spool_dir.resolve() != expected_spool:
            raise ValueError(f"notify.spool_dir must be {expected_spool}")

    return runbook
