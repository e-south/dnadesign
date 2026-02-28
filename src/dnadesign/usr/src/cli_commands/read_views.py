"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli_commands/read_views.py

Read-oriented command handlers and Parquet target helpers for USR CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..dataset import Dataset
from ..errors import SequencesError
from ..events import record_event
from ..io import read_parquet_head
from ..pretty import PrettyOpts, fmt_value
from ..ui import (
    print_df_plain,
    render_table_rich,
)
from . import read_parquet_targets


@dataclass(frozen=True)
class ReadViewDeps:
    is_explicit_path_target: Callable[[str | None], bool]
    exit_missing_path_target: Callable[[str], None]
    resolve_existing_dataset_id: Callable[[Path, str], str]
    resolve_dataset_name_interactive: Callable[[Path, str | None, bool], str | None]
    assert_not_legacy_dataset_path: Callable[[Path, Path | None], None]
    legacy_dataset_prefix: str
    legacy_dataset_path_error: str


def _resolve_parquet_from_dir(dir_path: Path, glob: str | None = None) -> Path:
    return read_parquet_targets._resolve_parquet_from_dir(dir_path, glob=glob)


def _resolve_parquet_target(path_like: Path, glob: str | None = None) -> Path:
    return read_parquet_targets._resolve_parquet_target(path_like, glob=glob)


def _select_parquet_target_interactive(
    path_like: Path,
    glob: str | None,
    use_rich: bool,
    *,
    deps: ReadViewDeps,
    root: Path | None = None,
    confirm_if_inferred: bool = False,
) -> Path | None:
    return read_parquet_targets._select_parquet_target_interactive(
        path_like,
        glob,
        use_rich,
        deps=deps,
        root=root,
        confirm_if_inferred=confirm_if_inferred,
    )


def _print_df(df: pd.DataFrame) -> None:
    print_df_plain(df)


def _pretty_opts_from(args) -> PrettyOpts:
    return PrettyOpts(
        max_colwidth=int(args.max_colwidth),
        max_list_items=int(args.max_list_items),
        precision=int(args.precision),
    )


def _pretty_df(df: pd.DataFrame, opts: PrettyOpts) -> pd.DataFrame:
    def _fmt_cell(x):
        return fmt_value(x, opts)

    if hasattr(df, "map"):
        return df.map(_fmt_cell)
    return df.applymap(_fmt_cell)


def _log_implicit_pick_if_dataset(
    pq_path: Path | None,
    reason: str,
    *,
    deps: ReadViewDeps,
) -> None:
    if pq_path is None:
        return
    p = Path(pq_path).resolve()
    d = p.parent
    if (d / "meta.md").exists():
        meta_path = d / "meta.md"
        lines = meta_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise SequencesError(f"meta.md is empty: {meta_path}")
        first = lines[0]
        if not first.startswith("name:"):
            raise SequencesError(f"meta.md missing leading name: line: {meta_path}")
        dataset_name = first.split(":", 1)[1].strip()
        if not dataset_name:
            raise SequencesError(f"meta.md has empty name field: {meta_path}")
        if dataset_name == deps.legacy_dataset_prefix or dataset_name.startswith(f"{deps.legacy_dataset_prefix}/"):
            raise SequencesError(deps.legacy_dataset_path_error)
        dataset_root = None
        parts = Path(dataset_name).parts
        if parts:
            try:
                dataset_root = d.parents[len(parts) - 1]
            except IndexError:
                dataset_root = None
        record_event(
            d / ".events.log",
            "implicit_file_pick",
            dataset=dataset_name,
            args={"path": str(p), "cwd": str(Path.cwd().resolve()), "reason": reason},
            target_path=p,
            dataset_root=dataset_root,
        )


def cmd_head(args, *, deps: ReadViewDeps) -> None:
    target = str(getattr(args, "target", "."))
    implicit = target in {"", ".", "./"}
    cols = [c.strip() for c in args.columns.split(",") if c.strip()] if args.columns else None
    if deps.is_explicit_path_target(target):
        p = Path(target).expanduser()
        if not p.exists():
            deps.exit_missing_path_target(target)
        pq_path = _select_parquet_target_interactive(
            p,
            glob=None,
            use_rich=bool(getattr(args, "rich", False)),
            deps=deps,
            root=args.root,
            confirm_if_inferred=implicit,
        )
        if pq_path is None:
            return
        if implicit:
            _log_implicit_pick_if_dataset(pq_path, reason="head", deps=deps)
        pf = pq.ParquetFile(str(pq_path))
        tbl = read_parquet_head(pq_path, int(args.n), columns=cols)
        df = tbl.to_pandas()
        caption = f"{pq_path}  rows={pf.metadata.num_rows:,}  cols={pf.metadata.num_columns}"
        if getattr(args, "rich", False):
            if not args.raw:
                df = _pretty_df(df, _pretty_opts_from(args))
            render_table_rich(
                df,
                title=str(pq_path),
                caption=caption,
                max_colwidth=int(args.max_colwidth),
            )
        else:
            print(f"# {caption}")
            if not args.raw:
                df = _pretty_df(df, _pretty_opts_from(args))
            _print_df(df)
        return
    ds_name = deps.resolve_existing_dataset_id(args.root, target)
    d = Dataset(args.root, ds_name)
    df = d.head(args.n, columns=cols, include_deleted=bool(getattr(args, "include_deleted", False)))
    meta = pq.ParquetFile(str(d.records_path)).metadata
    caption = f"{d.records_path}  rows={meta.num_rows:,}  cols={meta.num_columns}"
    if getattr(args, "rich", False):
        if not args.raw:
            df = _pretty_df(df, _pretty_opts_from(args))
        render_table_rich(
            df,
            title=f"dataset: {ds_name}",
            caption=caption,
            max_colwidth=int(args.max_colwidth),
        )
    else:
        if not args.raw:
            df = _pretty_df(df, _pretty_opts_from(args))
        _print_df(df)


def cmd_cols(args, *, deps: ReadViewDeps) -> None:
    path_arg = getattr(args, "path", None)
    target_arg = getattr(args, "target", None)
    tgt = str(path_arg or target_arg or ".")
    implicit = (path_arg is None) and tgt in {"", ".", "./"}
    pq_path: Path | None = None
    if path_arg is not None or deps.is_explicit_path_target(tgt):
        p = Path(tgt).expanduser()
        if not p.exists():
            deps.exit_missing_path_target(tgt)
        pq_path = _select_parquet_target_interactive(
            p,
            glob=args.glob,
            use_rich=bool(getattr(args, "rich", False)),
            deps=deps,
            root=args.root,
            confirm_if_inferred=implicit,
        )
        if pq_path is None:
            return
        if implicit:
            _log_implicit_pick_if_dataset(pq_path, reason="cols", deps=deps)
    else:
        ds_name = deps.resolve_existing_dataset_id(args.root, str(tgt))
        pq_path = args.root / ds_name / "records.parquet"
    if pq_path is None or not pq_path.exists():
        raise FileNotFoundError(f"Target not found: {tgt}")
    pf = pq.ParquetFile(str(pq_path))
    fields = []
    sch = pf.schema_arrow
    for i in range(len(sch.names)):
        f = sch.field(i)
        fields.append({"name": f.name, "type": str(f.type), "nullable": bool(f.nullable)})
    df = pd.DataFrame(fields, columns=["name", "type", "nullable"])
    caption = f"{pq_path}  rows={pf.metadata.num_rows:,}  cols={pf.metadata.num_columns}"
    if getattr(args, "rich", False):
        render_table_rich(df, title="columns", caption=caption)
    else:
        print(f"# {caption}")
        print_df_plain(df)


def cmd_describe(args, *, deps: ReadViewDeps) -> None:
    ds_name = deps.resolve_dataset_name_interactive(
        args.root, getattr(args, "dataset", None), bool(getattr(args, "rich", False))
    )
    if not ds_name:
        return
    d = Dataset(args.root, ds_name)
    cols = [c.strip() for c in args.columns.split(",") if c.strip()] if args.columns else None
    opts = PrettyOpts(
        max_colwidth=int(args.max_colwidth),
        max_list_items=int(args.max_list_items),
        precision=int(args.precision),
    )
    pf = pq.ParquetFile(str(d.records_path))
    prof = d.describe(
        opts,
        columns=cols,
        sample=int(args.sample),
        batch_size=65536,
        include_deleted=bool(getattr(args, "include_deleted", False)),
    )
    df = pd.DataFrame(
        prof,
        columns=[
            "column",
            "type",
            "non_null",
            "nulls",
            "null_pct",
            "example",
            "list_min",
            "list_max",
            "list_avg",
        ],
    )
    df["null_pct"] = df["null_pct"].map(lambda x: f"{x:.1f}%")
    if getattr(args, "rich", False):
        caption = f"{d.records_path}  rows={pf.metadata.num_rows:,}  cols={pf.metadata.num_columns}"
        render_table_rich(
            df,
            title=f"describe: {ds_name}",
            caption=caption,
            max_colwidth=int(args.max_colwidth),
        )
    else:
        print(f"# {d.records_path}  rows={pf.metadata.num_rows:,}  cols={pf.metadata.num_columns}")
        _print_df(df)


def cmd_cell(args, *, deps: ReadViewDeps) -> None:
    path_arg = getattr(args, "path", None)
    target_arg = getattr(args, "target", None)
    tgt = str(path_arg or target_arg or ".")
    implicit = (path_arg is None) and tgt in {"", ".", "./"}
    pq_path: Path | None = None
    if path_arg is not None or deps.is_explicit_path_target(tgt):
        p = Path(tgt).expanduser()
        if not p.exists():
            deps.exit_missing_path_target(tgt)
        pq_path = _select_parquet_target_interactive(
            p,
            glob=args.glob,
            use_rich=False,
            deps=deps,
            root=args.root,
            confirm_if_inferred=implicit,
        )
        if pq_path is None:
            return
        if implicit:
            _log_implicit_pick_if_dataset(pq_path, reason="cell", deps=deps)
    else:
        ds_name = deps.resolve_existing_dataset_id(args.root, str(tgt))
        pq_path = args.root / ds_name / "records.parquet"
    if pq_path is None or not Path(pq_path).exists():
        raise FileNotFoundError(f"Target not found: {tgt}")
    col = str(args.col)
    row = int(args.row)
    try:
        tbl = pq.read_table(pq_path, columns=[col])
    except (KeyError, pa.ArrowInvalid):
        pf = pq.ParquetFile(str(pq_path))
        names = ", ".join(pf.schema_arrow.names)
        raise SequencesError(f"Column '{col}' not found in {pq_path}. Available columns: {names}")
    if row < 0 or row >= tbl.num_rows:
        raise SequencesError(f"Row {row} out of range (0..{tbl.num_rows - 1}).")
    val = tbl.column(0)[row].as_py()
    print(val)
