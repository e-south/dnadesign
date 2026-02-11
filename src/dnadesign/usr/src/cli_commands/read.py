"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli_commands/read.py

USR CLI read-only command implementations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import datetime as dt
from typing import Callable

import pandas as pd
import pyarrow.parquet as pq

from ..dataset import Dataset
from ..pretty import render_schema_tree
from ..ui import print_df_plain, render_schema_tree_rich, render_table_rich
from .datasets import list_datasets, resolve_dataset_name_interactive


def cmd_ls(
    args,
    *,
    resolve_output_format: Callable[[object], str],
    print_json: Callable[[dict], None],
    output_version: int,
) -> None:
    fmt = resolve_output_format(args)
    names = list_datasets(args.root)
    if not names:
        print(f"(no datasets under {args.root})")
        return

    def _fmt_bytes(size: int | None) -> str:
        if not isinstance(size, int):
            return "?"
        units = ["B", "KB", "MB", "GB", "TB", "PB"]
        i = 0
        x = float(size)
        while x >= 1024 and i < len(units) - 1:
            x /= 1024.0
            i += 1
        return f"{x:.0f}{units[i]}"

    rows = []
    for name in names:
        records_path = args.root / name / "records.parquet"
        pf = pq.ParquetFile(str(records_path))
        stats = records_path.stat()
        rows.append(
            {
                "dataset": name,
                "rows": pf.metadata.num_rows,
                "cols": pf.metadata.num_columns,
                "size_bytes": int(stats.st_size),
                "updated": dt.datetime.fromtimestamp(int(stats.st_mtime)).isoformat(timespec="seconds"),
            }
        )

    if fmt == "json":
        print_json({"usr_output_version": output_version, "data": rows})
        return

    table_rows = []
    for row in rows:
        table_rows.append(
            {
                "dataset": row["dataset"],
                "rows": row["rows"],
                "cols": row["cols"],
                "size": _fmt_bytes(int(row["size_bytes"])),
                "updated": row["updated"],
            }
        )

    df = pd.DataFrame(table_rows, columns=["dataset", "rows", "cols", "size", "updated"])
    if fmt == "rich":
        render_table_rich(df, title="USR datasets", caption=str(args.root))
    else:
        print_df_plain(df)


def cmd_info(
    args,
    *,
    resolve_output_format: Callable[[object], str],
    print_json: Callable[[dict], None],
    output_version: int,
    resolve_dataset_for_read: Callable[[object, str], Dataset],
) -> None:
    fmt = resolve_output_format(args)
    dataset_arg = getattr(args, "dataset", None)
    if not dataset_arg:
        ds_name = resolve_dataset_name_interactive(args.root, dataset_arg, bool(getattr(args, "rich", False)))
        if not ds_name:
            return
        dataset = Dataset(args.root, ds_name)
    else:
        dataset = resolve_dataset_for_read(args.root, str(dataset_arg))
    info = dataset.info_dict()
    if fmt == "json":
        print_json({"usr_output_version": output_version, "data": info})
        return
    for key, value in info.items():
        print(f"{key}: {value}")


def cmd_schema(
    args,
    *,
    resolve_output_format: Callable[[object], str],
    print_json: Callable[[dict], None],
    output_version: int,
) -> None:
    fmt = resolve_output_format(args)
    ds_name = resolve_dataset_name_interactive(
        args.root,
        getattr(args, "dataset", None),
        bool(getattr(args, "rich", False)),
    )
    if not ds_name:
        return
    dataset = Dataset(args.root, ds_name)
    schema = dataset.schema()
    if fmt == "json":
        fields = [{"name": field.name, "type": str(field.type), "nullable": field.nullable} for field in schema]
        print_json({"usr_output_version": output_version, "data": {"fields": fields}})
        return
    if args.tree:
        if fmt == "rich":
            lines = render_schema_tree(schema).splitlines()
            render_schema_tree_rich(lines, title=str(dataset.records_path))
        else:
            print(render_schema_tree(schema))
        return
    print(schema)
