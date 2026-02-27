"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli_commands/maintenance.py

Maintenance command handlers for USR datasets.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..dataset import Dataset
from ..errors import SequencesError


@dataclass(frozen=True)
class MaintenanceDeps:
    resolve_dataset_name_interactive: Callable[[Path, str | None, bool], str | None]
    prompt: Callable[[str], str]


def cmd_registry_freeze(args, *, deps: MaintenanceDeps) -> None:
    ds_name = deps.resolve_dataset_name_interactive(args.root, getattr(args, "dataset", None), False)
    if not ds_name:
        return
    dataset = Dataset(args.root, ds_name)
    with dataset.maintenance(reason="registry_freeze"):
        snap = dataset.freeze_registry()
    print(f"[registry-freeze] wrote {snap}")


def cmd_overlay_compact(args, *, deps: MaintenanceDeps) -> None:
    ds_name = deps.resolve_dataset_name_interactive(args.root, getattr(args, "dataset", None), False)
    if not ds_name:
        return
    namespace = getattr(args, "namespace", None)
    if not namespace:
        raise SequencesError("overlay-compact requires a namespace argument.")
    dataset = Dataset(args.root, ds_name)
    with dataset.maintenance(reason="overlay_compact"):
        out_path = dataset.compact_overlay(str(namespace))
    print(f"[overlay-compact] wrote {out_path}")


def cmd_snapshot(args, *, deps: MaintenanceDeps) -> None:
    ds_name = deps.resolve_dataset_name_interactive(args.root, getattr(args, "dataset", None), False)
    if not ds_name:
        return
    dataset = Dataset(args.root, ds_name)
    dataset.snapshot()
    print(f"Snapshot saved under {dataset.snapshot_dir}")
    dataset.append_meta_note("Snapshot saved", f"usr snapshot {ds_name}")


def cmd_dedupe_sequences(args, *, deps: MaintenanceDeps) -> None:
    dataset = Dataset(args.root, args.dataset)
    with dataset.maintenance(reason="dedupe"):
        stats = dataset.dedupe(
            key=str(args.key),
            keep=str(args.keep),
            batch_size=int(args.batch_size),
            dry_run=True,
        )
    if stats.rows_dropped == 0:
        print("OK: no duplicate keys found.")
        return
    print(f"Found {stats.groups} duplicate group(s); would drop {stats.rows_dropped} row(s).")
    if args.dry_run:
        return
    if not args.yes:
        ans = deps.prompt("Proceed with destructive de-duplication? [y/N]: ").strip().lower()
        if ans not in {"y", "yes"}:
            print("Aborted.")
            return
    with dataset.maintenance(reason="dedupe"):
        stats = dataset.dedupe(
            key=str(args.key),
            keep=str(args.keep),
            batch_size=int(args.batch_size),
            dry_run=False,
        )
    rows_after = stats.rows_total - stats.rows_dropped
    print(f"[dedupe] dropped {stats.rows_dropped} row(s); dataset now has {rows_after} rows.")
