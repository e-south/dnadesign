"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli_commands/state.py

USR CLI state and soft-delete command implementations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Callable

from ..dataset import Dataset
from ..ui import print_df_plain, render_table_rich
from .datasets import resolve_dataset_name_interactive


def cmd_delete(
    args,
    *,
    collect_ids: Callable[[list[str] | None, object], list[str]],
) -> None:
    ds_name = resolve_dataset_name_interactive(args.root, getattr(args, "dataset", None), False)
    if not ds_name:
        return
    dataset = Dataset(args.root, ds_name)
    ids = collect_ids(getattr(args, "id", None), getattr(args, "id_file", None))
    count = dataset.tombstone(
        ids,
        reason=getattr(args, "reason", None),
        allow_missing=bool(getattr(args, "allow_missing", False)),
    )
    print(f"Tombstoned {count} record(s) in {dataset.name}")


def cmd_restore(
    args,
    *,
    collect_ids: Callable[[list[str] | None, object], list[str]],
) -> None:
    ds_name = resolve_dataset_name_interactive(args.root, getattr(args, "dataset", None), False)
    if not ds_name:
        return
    dataset = Dataset(args.root, ds_name)
    ids = collect_ids(getattr(args, "id", None), getattr(args, "id_file", None))
    count = dataset.restore(ids, allow_missing=bool(getattr(args, "allow_missing", False)))
    print(f"Restored {count} record(s) in {dataset.name}")


def cmd_state_set(
    args,
    *,
    collect_ids: Callable[[list[str] | None, object], list[str]],
    collect_list: Callable[[list[str] | None], list[str]],
) -> None:
    ds_name = resolve_dataset_name_interactive(args.root, getattr(args, "dataset", None), False)
    if not ds_name:
        return
    dataset = Dataset(args.root, ds_name)
    ids = collect_ids(getattr(args, "id", None), getattr(args, "id_file", None))
    lineage = collect_list(getattr(args, "lineage", None))
    rows = dataset.set_state(
        ids,
        masked=getattr(args, "masked", None),
        qc_status=getattr(args, "qc_status", None) or None,
        split=getattr(args, "split", None) or None,
        supersedes=getattr(args, "supersedes", None) or None,
        lineage=lineage or None,
        allow_missing=bool(getattr(args, "allow_missing", False)),
    )
    print(f"Updated usr_state for {rows} record(s) in {dataset.name}")


def cmd_state_clear(
    args,
    *,
    collect_ids: Callable[[list[str] | None, object], list[str]],
) -> None:
    ds_name = resolve_dataset_name_interactive(args.root, getattr(args, "dataset", None), False)
    if not ds_name:
        return
    dataset = Dataset(args.root, ds_name)
    ids = collect_ids(getattr(args, "id", None), getattr(args, "id_file", None))
    rows = dataset.clear_state(ids, allow_missing=bool(getattr(args, "allow_missing", False)))
    print(f"Cleared usr_state for {rows} record(s) in {dataset.name}")


def cmd_state_get(
    args,
    *,
    collect_ids: Callable[[list[str] | None, object], list[str]],
    resolve_output_format: Callable[[object], str],
    print_json: Callable[[dict], None],
    output_version: int,
) -> None:
    ds_name = resolve_dataset_name_interactive(args.root, getattr(args, "dataset", None), False)
    if not ds_name:
        return
    dataset = Dataset(args.root, ds_name)
    ids = collect_ids(getattr(args, "id", None), getattr(args, "id_file", None))
    df = dataset.get_state(ids, allow_missing=bool(getattr(args, "allow_missing", False)))
    fmt = resolve_output_format(args)
    if fmt == "json":
        print_json({"usr_output_version": output_version, "data": df.to_dict(orient="records")})
        return
    if fmt == "rich":
        render_table_rich(df, title=f"usr_state: {dataset.name}")
        return
    print_df_plain(df)
