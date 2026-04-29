"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli/commands/maintenance/events.py

USR CLI maintenance command implementations for dataset event logs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ....dataset import Dataset
from ....events import garden_event_log
from .registry import MaintenanceDeps


def cmd_event_log_garden(args, *, deps: MaintenanceDeps) -> None:
    ds_name = deps.resolve_dataset_name_interactive(args.root, getattr(args, "dataset", None), False)
    if not ds_name:
        return
    dataset = Dataset(args.root, ds_name)
    result = garden_event_log(
        dataset,
        retain_last=int(getattr(args, "retain_last", 1000)),
        write=bool(getattr(args, "write", False)),
        acknowledge_notify_cursor_reset=bool(getattr(args, "acknowledge_notify_cursor_reset", False)),
        reason=str(getattr(args, "reason", "") or ""),
    )
    print(
        "[event-log-garden] "
        f"mode={result.mode} dataset={result.dataset_id} total_lines={result.total_lines} "
        f"retained_lines={result.retained_lines} archived_lines={result.archived_lines} "
        f"archive={result.archive_path or ''}"
    )
