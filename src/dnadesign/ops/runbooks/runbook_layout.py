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
        infer_config = runbook.infer.config.resolve()
        if not _is_path_within(path=infer_config, parent=workspace_root):
            raise ValueError(f"infer.config must be within {workspace_root}")
        if infer_config.suffix.lower() not in {".yaml", ".yml"} or not infer_config.name.startswith("config"):
            raise ValueError(
                "infer.config must point to a workspace-local YAML config whose filename starts with 'config'"
            )
    if runbook.notify is not None:
        notify_root = (workspace_root / "outputs" / "notify" / runbook.notify.tool).resolve()
        profile_path = runbook.notify.profile.resolve()
        cursor_path = runbook.notify.cursor.resolve()
        spool_path = runbook.notify.spool_dir.resolve()
        if not _is_path_within(path=profile_path, parent=notify_root):
            raise ValueError(f"notify.profile must be within {notify_root}")
        if not _is_path_within(path=cursor_path, parent=notify_root):
            raise ValueError(f"notify.cursor must be within {notify_root}")
        if not _is_path_within(path=spool_path, parent=notify_root):
            raise ValueError(f"notify.spool_dir must be within {notify_root}")
        if profile_path.name != "profile.json":
            raise ValueError("notify.profile filename must be exactly profile.json")
        if cursor_path.name != "cursor":
            raise ValueError("notify.cursor filename must be exactly cursor")
        if spool_path.name != "spool":
            raise ValueError("notify.spool_dir directory name must be exactly spool")
        if profile_path.parent != cursor_path.parent or profile_path.parent != spool_path.parent:
            raise ValueError("notify.profile, notify.cursor, and notify.spool_dir must share the same lane directory")

    return runbook
