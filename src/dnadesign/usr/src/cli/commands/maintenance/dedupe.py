"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli/commands/maintenance/dedupe.py

USR CLI maintenance dedupe command implementations.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ....dataset import Dataset
from .registry import MaintenanceDeps


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
