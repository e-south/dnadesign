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


@dataclass(frozen=True)
class ReadViewDeps:
    is_explicit_path_target: Callable[[str | None], bool]
    exit_missing_path_target: Callable[[str], None]
    resolve_existing_dataset_id: Callable[[Path, str], str]
    resolve_dataset_name_interactive: Callable[[Path, str | None, bool], str | None]
    assert_not_legacy_dataset_path: Callable[[Path, Path | None], None]
    legacy_dataset_prefix: str
    legacy_dataset_path_error: str


def _list_parquet_candidates(dir_path: Path, glob: str | None = None) -> list[Path]:
    dir_path = Path(dir_path)
    if not dir_path.exists() or not dir_path.is_dir():
        return []
    seen: dict[Path, None] = {}

    def _add(p: Path) -> None:
        if p.exists() and p.is_file() and p.suffix.lower() == ".parquet":
            seen[p.resolve()] = None

    if glob:
        for p in sorted(dir_path.glob(glob)):
            _add(p)
    _add(dir_path / "records.parquet")
    _add(dir_path / "events.parquet")
    for p in sorted(dir_path.glob("events*.parquet")):
        _add(p)
    for p in sorted(dir_path.glob("*.parquet")):
        _add(p)
    cands = list(seen.keys())
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands


def _human_size(n: int | None) -> str:
    if not isinstance(n, int):
        return "?"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    x = float(n)
    while x >= 1024 and i < len(units) - 1:
        x /= 1024.0
        i += 1
    return f"{x:.0f}{units[i]}"


def _prompt_pick_parquet(cands: list[Path], use_rich: bool) -> Path | None:
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    rows = []
    for idx, p in enumerate(cands, start=1):
        pf = pq.ParquetFile(str(p))
        rows.append(
            {
                "#": idx,
                "file": p.name,
                "rows": pf.metadata.num_rows,
                "cols": pf.metadata.num_columns,
                "size": _human_size(int(p.stat().st_size)),
            }
        )
    df = pd.DataFrame(rows, columns=["#", "file", "rows", "cols", "size"])
    msg = "Multiple Parquet files found. Choose one by number (Enter = newest, q = abort):"
    if use_rich:
        render_table_rich(df, title="Pick a Parquet file", caption=msg)
    else:
        print_df_plain(df)
        print(msg)
    sel = input("> ").strip().lower()
    if sel in {"q", "quit", "n"}:
        print("Aborted.")
        return None
    if not sel:
        return cands[0]
    try:
        k = int(sel)
        if 1 <= k <= len(cands):
            return cands[k - 1]
    except ValueError:
        pass
    print("Invalid selection. Aborted.")
    return None


def _select_parquet_target_interactive(
    path_like: Path,
    glob: str | None,
    use_rich: bool,
    *,
    deps: ReadViewDeps,
    root: Path | None = None,
    confirm_if_inferred: bool = False,
) -> Path | None:
    _ = confirm_if_inferred
    p = Path(path_like)
    if p.exists():
        deps.assert_not_legacy_dataset_path(p, root)
    if p.is_file() and p.suffix.lower() == ".parquet":
        return p
    if p.is_dir():
        cands = _list_parquet_candidates(p, glob=glob)
        if not cands:
            print(f"(no Parquet files found under {p})")
            print(
                "Tip: cd into a dataset folder (with records.parquet) or pass a dataset name (e.g., 'usr cols demo')."
            )
            return None
        if len(cands) == 1:
            return cands[0]
        return _prompt_pick_parquet(cands, use_rich)
    return None


def _resolve_parquet_from_dir(dir_path: Path, glob: str | None = None) -> Path:
    dir_path = Path(dir_path)
    if not dir_path.exists() or not dir_path.is_dir():
        raise FileNotFoundError(f"{dir_path} is not a directory")
    cands = []
    if (dir_path / "events.parquet").exists():
        return dir_path / "events.parquet"
    if glob:
        cands = sorted(dir_path.glob(glob))
    if not cands:
        cands = sorted(dir_path.glob("events*.parquet"))
    if not cands and (dir_path / "records.parquet").exists():
        return dir_path / "records.parquet"
    if not cands:
        cands = sorted(dir_path.glob("*.parquet"))
    if not cands:
        raise FileNotFoundError(f"No Parquet files found under {dir_path}")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _resolve_parquet_target(path_like: Path, glob: str | None = None) -> Path:
    p = Path(path_like)
    if p.is_file() and p.suffix.lower() == ".parquet":
        return p
    if p.is_dir():
        return _resolve_parquet_from_dir(p, glob=glob)
    raise FileNotFoundError(f"Target not found: {p}")


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
