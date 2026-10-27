"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli/commands/lifecycle/snapshot.py

USR CLI lifecycle snapshot command implementation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ....dataset import Dataset


@dataclass(frozen=True)
class SnapshotDeps:
    resolve_dataset_name_interactive: Callable[[Path, str | None, bool], str | None]


def cmd_snapshot(args, *, deps: SnapshotDeps) -> None:
    ds_name = deps.resolve_dataset_name_interactive(args.root, getattr(args, "dataset", None), False)
    if not ds_name:
        return
    dataset = Dataset(args.root, ds_name)
    dataset.snapshot()
    print(f"Snapshot saved under {dataset.snapshot_dir}")
    dataset.append_meta_note("Snapshot saved", f"usr snapshot {ds_name}")
