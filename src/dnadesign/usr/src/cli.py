"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/src/cli.py

Typer CLI entrypoint for USR dataset operations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace as NS

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import typer

from .cli_commands import datasets as dataset_commands
from .cli_commands import read as read_commands
from .cli_commands import remotes as remotes_commands
from .cli_commands import state as state_commands
from .cli_commands import sync as sync_commands
from .cli_commands import write as write_commands
from .cli_event_output import emit_event_line as _emit_event_line_value
from .cli_merge_policy import resolve_merge_policy
from .cli_paths import (
    LEGACY_DATASET_PATH_ERROR as _LEGACY_DATASET_PATH_ERROR,
)
from .cli_paths import (
    assert_not_legacy_dataset_path as _assert_not_legacy_dataset_path_impl,
)
from .cli_paths import (
    assert_supported_root as _assert_supported_root_impl,
)
from .cli_paths import (
    pkg_usr_root as _pkg_usr_root_impl,
)
from .cli_paths import (
    resolve_dataset_for_read as _resolve_dataset_for_read_impl,
)
from .cli_paths import (
    resolve_path_anywhere as _resolve_path_anywhere_impl,
)
from .dataset import LEGACY_DATASET_PREFIX, Dataset
from .errors import DuplicateIDError, SequencesError, UserAbort
from .events import record_event
from .io import read_parquet_head
from .merge_datasets import (
    MergeColumnsMode,
    merge_usr_to_usr,
)
from .mock import MockSpec, add_demo_columns, create_mock_dataset
from .overlays import list_overlays, overlay_metadata
from .pretty import PrettyOpts, fmt_value
from .registry import load_registry, parse_columns_spec, register_namespace
from .ui import (
    print_df_plain,
    render_table_rich,
)

# Compatibility exports kept for existing monkeypatch-based tests.
shutil = remotes_commands.shutil
SSHRemote = remotes_commands.SSHRemote


# --------- small helpers for path-first UX (no-traceback, interactive pick) ---------
def _list_parquet_candidates(dir_path: Path, glob: str | None = None) -> list[Path]:
    dir_path = Path(dir_path)
    if not dir_path.exists() or not dir_path.is_dir():
        return []
    seen: dict[Path, None] = {}

    def _add(p: Path) -> None:
        if p.exists() and p.is_file() and p.suffix.lower() == ".parquet":
            seen[p.resolve()] = None

    # Optional explicit glob first
    if glob:
        for p in sorted(dir_path.glob(glob)):
            _add(p)
    # Prefer canonical files
    _add(dir_path / "records.parquet")
    _add(dir_path / "events.parquet")
    # Common patterns, then anything else
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


USR_OUTPUT_VERSION = 1
LEGACY_DATASET_PATH_ERROR = _LEGACY_DATASET_PATH_ERROR


def _resolve_output_format(args, *, default: str = "auto") -> str:
    fmt = str(getattr(args, "format", default) or default).lower()
    if fmt not in {"auto", "rich", "plain", "json"}:
        raise SequencesError(f"Unsupported format '{fmt}'. Use auto|rich|plain|json.")
    if fmt == "auto":
        if _is_interactive() and bool(getattr(args, "rich", True)):
            return "rich"
        return "plain"
    return fmt


def _print_json(payload) -> None:
    print(json.dumps(payload, separators=(",", ":")))


def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _prompt_pick_parquet(cands: list[Path], use_rich: bool) -> Path | None:
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    # Build a compact table with mtime/rows/cols
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
        return cands[0]  # newest by mtime
    try:
        k = int(sel)
        if 1 <= k <= len(cands):
            return cands[k - 1]
    except ValueError:
        pass
    print("Invalid selection. Aborted.")
    return None


def cmd_repair_densegen(args):
    from .convert_legacy import repair_densegen_used_tfbs

    ds_name = _resolve_dataset_name_interactive(
        args.root, getattr(args, "dataset", None), bool(getattr(args, "rich", False))
    )
    if not ds_name:
        return
    stats = repair_densegen_used_tfbs(
        dataset_root=args.root,
        dataset_name=ds_name,
        min_tfbs_len=int(getattr(args, "min_tfbs_len", 6)),
        dry_run=bool(getattr(args, "dry_run", False)),
        assume_yes=bool(getattr(args, "yes", False)),
        dedupe_policy=(None if getattr(args, "dedupe", "off") == "off" else getattr(args, "dedupe")),
        drop_missing_used_tfbs=bool(getattr(args, "drop_missing_used_tfbs", False)),
        drop_single_tf=bool(getattr(args, "drop_single_tf", False)),
        drop_id_seq_only=bool(getattr(args, "drop_id_seq_only", False)),
        filter_single_tf=bool(getattr(args, "filter_single_tf", False)),
    )
    print(
        f"[repair-densegen] rows={stats.rows_total}  touched={stats.rows_touched}  "
        f"changed(parts/used/detail/counts/u_list)={stats.rows_changed_tfbs_parts}/"
        f"{stats.rows_changed_used_tfbs}/{stats.rows_changed_used_detail}/"
        f"{stats.rows_changed_used_counts}/{stats.rows_changed_used_list}"
    )


def _select_parquet_target_interactive(
    path_like: Path,
    glob: str | None,
    use_rich: bool,
    root: Path | None = None,
    confirm_if_inferred: bool = False,
) -> Path | None:
    p = Path(path_like)
    if p.exists():
        _assert_not_legacy_dataset_path(p, root=root)
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
        # If exactly one Parquet file is present, auto-select it with no prompt.
        # Only present an interactive picker when there are multiple viable files.
        if len(cands) == 1:
            return cands[0]
        return _prompt_pick_parquet(cands, use_rich)
    # Not a file or directory → let caller handle dataset/other cases
    return None


# --------- dataset guessing (path-first) ----------
def _normalize_dataset_id(dataset: str) -> str:
    return dataset_commands._normalize_dataset_id(dataset)  # noqa: SLF001


def _resolve_existing_dataset_id(root: Path, dataset: str) -> str:
    return dataset_commands.resolve_existing_dataset_id(root, dataset)


def _resolve_dataset_name_interactive(root: Path, dataset: str | None, use_rich: bool) -> str | None:
    return dataset_commands.resolve_dataset_name_interactive(root, dataset, use_rich)


def _is_explicit_path_target(target: str | None) -> bool:
    text = str(target or "").strip()
    if text in {"", ".", "./", "..", "../"}:
        return True
    if text.startswith("./") or text.startswith("../") or text.startswith("~/"):
        return True
    if Path(text).is_absolute():
        return True
    if text.lower().endswith(".parquet"):
        return True
    if "/" in text or "\\" in text:
        return Path(text).expanduser().exists()
    return False


def _exit_missing_path_target(target: str) -> None:
    print(f"ERROR: Path target not found: {target}")
    raise typer.Exit(code=4)


def _resolve_dataset_for_read(root: Path, dataset_arg: str) -> Dataset:
    return _resolve_dataset_for_read_impl(
        root,
        dataset_arg,
        resolve_existing_dataset_id=_resolve_existing_dataset_id,
        normalize_dataset_id=_normalize_dataset_id,
        pkg_root=_pkg_usr_root(),
    )


def _assert_not_legacy_dataset_path(path: Path, *, root: Path | None = None) -> None:
    _assert_not_legacy_dataset_path_impl(path, root=root, pkg_root=_pkg_usr_root())


# ---------------- path helpers: resolve paths relative to the installed package ----------------


def _pkg_usr_root() -> Path:
    """
    Return the installed dnadesign/usr package directory.
    This is stable no matter where the user runs 'usr' from.
    """
    return _pkg_usr_root_impl()


def _assert_supported_root(root: Path) -> None:
    _assert_supported_root_impl(root, pkg_root=_pkg_usr_root())


def _resolve_path_anywhere(p: Path) -> Path:
    """
    Make file arguments robust:
      1) absolute path → as-is
      2) relative path existing under CWD → as-is
      3) otherwise, try to resolve relative to the installed dnadesign/usr package,
         including common repo-style prefixes like 'src/dnadesign/usr/...'
         or 'usr/...'.
    """
    return _resolve_path_anywhere_impl(p, pkg_root=_pkg_usr_root())


# ---------- helpers & command impls ----------
def list_datasets(root: Path):
    return dataset_commands.list_datasets(root)


def cmd_ls(args):
    read_commands.cmd_ls(
        args,
        resolve_output_format=_resolve_output_format,
        print_json=_print_json,
        output_version=USR_OUTPUT_VERSION,
    )


def cmd_init(args):
    write_commands.cmd_init(args)


def cmd_import(args):
    write_commands.cmd_import(
        args,
        resolve_path_anywhere=_resolve_path_anywhere,
    )


def cmd_attach(args):
    write_commands.cmd_attach(
        args,
        resolve_path_anywhere=_resolve_path_anywhere,
    )


# ---------------- error formatting (actionable nudges) ----------------
def _abbrev_id(rid: str, n: int = 8) -> str:
    return (rid[:n] + "…") if isinstance(rid, str) and len(rid) > n else rid


def _print_dup_groups(title: str, groups) -> None:
    if not groups:
        return
    print(title)
    for g in groups:
        rid = _abbrev_id(g.id)
        rows = ",".join(str(i) for i in g.rows) if g.rows else "n/a"
        seq = g.sequence
        print(f"  • id={rid:<10}  count={g.count:<3}  input rows=[{rows}]")
        print(f"    sequence: {seq}")


def _print_user_error(e: SequencesError) -> None:
    """
    Consistent, succinct, user-facing error rendering.
    """
    print(f"ERROR: {e}")

    if isinstance(e, DuplicateIDError):
        if getattr(e, "groups", None):
            _print_dup_groups("Top duplicates (showing up to 5):", e.groups[:5])
        if getattr(e, "casefold_groups", None):
            _print_dup_groups(
                "Case-insensitive duplicates (same letters, different capitalization):",
                e.casefold_groups[:5],
            )
        if getattr(e, "hint", None):
            print(f"Hint: {e.hint}")
        return

    hint = getattr(e, "hint", None)
    if hint:
        print(f"Hint: {hint}")


def cmd_info(args):
    read_commands.cmd_info(
        args,
        resolve_output_format=_resolve_output_format,
        print_json=_print_json,
        output_version=USR_OUTPUT_VERSION,
        resolve_dataset_for_read=_resolve_dataset_for_read,
    )


def cmd_schema(args):
    read_commands.cmd_schema(
        args,
        resolve_output_format=_resolve_output_format,
        print_json=_print_json,
        output_version=USR_OUTPUT_VERSION,
    )


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


def _log_implicit_pick_if_dataset(pq_path: Path | None, reason: str) -> None:
    """
    If the selected Parquet file lives inside a USR dataset folder (has meta.md),
    append a small event so the implicit behavior is auditable.
    """
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
        if dataset_name == LEGACY_DATASET_PREFIX or dataset_name.startswith(f"{LEGACY_DATASET_PREFIX}/"):
            raise SequencesError(LEGACY_DATASET_PATH_ERROR)
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


def cmd_head(args):
    target = str(getattr(args, "target", "."))
    implicit = target in {"", ".", "./"}
    cols = [c.strip() for c in args.columns.split(",") if c.strip()] if args.columns else None
    if _is_explicit_path_target(target):
        p = Path(target).expanduser()
        if not p.exists():
            _exit_missing_path_target(target)
        pq_path = _select_parquet_target_interactive(
            p,
            glob=None,
            use_rich=bool(getattr(args, "rich", False)),
            root=args.root,
            confirm_if_inferred=implicit,
        )
        if pq_path is None:
            return
        if implicit:
            _log_implicit_pick_if_dataset(pq_path, reason="head")
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
    # dataset mode for plain IDs (no implicit cwd path fallback)
    ds_name = _resolve_existing_dataset_id(args.root, target)
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


def cmd_cols(args):
    path_arg = getattr(args, "path", None)
    target_arg = getattr(args, "target", None)
    tgt = str(path_arg or target_arg or ".")
    implicit = (path_arg is None) and tgt in {"", ".", "./"}
    pq_path: Path | None = None
    if path_arg is not None or _is_explicit_path_target(tgt):
        p = Path(tgt).expanduser()
        if not p.exists():
            _exit_missing_path_target(tgt)
        pq_path = _select_parquet_target_interactive(
            p,
            glob=args.glob,
            use_rich=bool(getattr(args, "rich", False)),
            root=args.root,
            confirm_if_inferred=implicit,
        )
        if pq_path is None:
            return
        if implicit:
            _log_implicit_pick_if_dataset(pq_path, reason="cols")
    else:
        ds_name = _resolve_existing_dataset_id(args.root, str(tgt))
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


def cmd_describe(args):
    ds_name = _resolve_dataset_name_interactive(
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


def cmd_cell(args):
    # Determine target path: --path, or positional, or dataset name
    path_arg = getattr(args, "path", None)
    target_arg = getattr(args, "target", None)
    tgt = str(path_arg or target_arg or ".")
    implicit = (path_arg is None) and tgt in {"", ".", "./"}
    pq_path: Path | None = None
    if path_arg is not None or _is_explicit_path_target(tgt):
        p = Path(tgt).expanduser()
        if not p.exists():
            _exit_missing_path_target(tgt)
        pq_path = _select_parquet_target_interactive(
            p,
            glob=args.glob,
            use_rich=False,
            root=args.root,
            confirm_if_inferred=implicit,
        )
        if pq_path is None:
            return
        if implicit:
            _log_implicit_pick_if_dataset(pq_path, reason="cell")
    else:
        ds_name = _resolve_existing_dataset_id(args.root, str(tgt))
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


def cmd_validate(args):
    dataset_arg = getattr(args, "dataset", None)
    if dataset_arg:
        d = _resolve_dataset_for_read(args.root, str(dataset_arg))
    else:
        ds_name = _resolve_dataset_name_interactive(args.root, dataset_arg, False)
        if not ds_name:
            return
        d = Dataset(args.root, ds_name)
    d.validate(
        strict=bool(getattr(args, "strict", False)),
        registry_mode=str(getattr(args, "registry_mode", "current")),
    )
    print("OK: validation passed.")


def cmd_registry_freeze(args) -> None:
    ds_name = _resolve_dataset_name_interactive(args.root, getattr(args, "dataset", None), False)
    if not ds_name:
        return
    d = Dataset(args.root, ds_name)
    with d.maintenance(reason="registry_freeze"):
        snap = d.freeze_registry()
    print(f"[registry-freeze] wrote {snap}")


def cmd_overlay_compact(args) -> None:
    ds_name = _resolve_dataset_name_interactive(args.root, getattr(args, "dataset", None), False)
    if not ds_name:
        return
    namespace = getattr(args, "namespace", None)
    if not namespace:
        raise SequencesError("overlay-compact requires a namespace argument.")
    d = Dataset(args.root, ds_name)
    with d.maintenance(reason="overlay_compact"):
        out_path = d.compact_overlay(str(namespace))
    print(f"[overlay-compact] wrote {out_path}")


def _emit_event_line(line: str, fmt: str) -> None:
    emitted = _emit_event_line_value(line, fmt)
    if emitted is not None:
        print(emitted)


def cmd_events_tail(args) -> None:
    dataset_arg = getattr(args, "dataset", None)
    if dataset_arg:
        d = _resolve_dataset_for_read(args.root, str(dataset_arg))
    else:
        ds_name = _resolve_dataset_name_interactive(args.root, dataset_arg, False)
        if not ds_name:
            return
        d = Dataset(args.root, ds_name)
    events_path = d.events_path
    if not events_path.exists():
        raise SequencesError(f"Events log not found: {events_path}")

    fmt = str(getattr(args, "format", "json")).strip().lower()
    n = int(getattr(args, "n", 0))
    follow = bool(getattr(args, "follow", False))

    if n > 0:
        tail_lines: deque[str] = deque(maxlen=n)
        with events_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                tail_lines.append(line)
        for line in tail_lines:
            _emit_event_line(line, fmt)
    else:
        with events_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                _emit_event_line(line, fmt)

    if not follow:
        return

    with events_path.open("r", encoding="utf-8") as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if line:
                _emit_event_line(line, fmt)
                continue
            time.sleep(0.2)


def cmd_get(args):
    ds_name = _resolve_dataset_name_interactive(
        args.root, getattr(args, "dataset", None), bool(getattr(args, "rich", False))
    )
    if not ds_name:
        return
    rid = getattr(args, "id", None) or getattr(args, "id_positional", None)
    if not rid:
        print("Usage: usr get [dataset] --id <sha1>  (or)  usr get <sha1>")
        return
    d = Dataset(args.root, ds_name)
    cols = [c.strip() for c in args.columns.split(",")] if args.columns else None
    df = d.get(rid, columns=cols, include_deleted=bool(getattr(args, "include_deleted", False)))
    if df.empty:
        print("Not found.")
    elif getattr(args, "rich", False):
        df_fmt = df.applymap(lambda x: fmt_value(x, PrettyOpts()))
        render_table_rich(df_fmt, title=f"record: {rid}", caption=str(d.records_path))
    else:
        _print_df(df)


def cmd_grep(args):
    ds_name = _resolve_dataset_name_interactive(
        args.root, getattr(args, "dataset", None), bool(getattr(args, "rich", False))
    )
    if not ds_name:
        return
    d = Dataset(args.root, ds_name)
    df = d.grep(
        args.pattern,
        args.limit,
        batch_size=int(args.batch_size),
        include_deleted=bool(getattr(args, "include_deleted", False)),
    )
    if getattr(args, "rich", False):
        df_fmt = df.applymap(lambda x: fmt_value(x, PrettyOpts()))
        render_table_rich(df_fmt, title=f"grep: {args.pattern}")
    else:
        _print_df(df)


def _default_export_filename(dataset_name: str, fmt: str) -> str:
    stem = Path(dataset_name).as_posix().strip("/").replace("/", "_")
    return f"{stem}.{fmt}"


def _resolve_export_target(out_path: Path, *, dataset_name: str, fmt: str) -> Path:
    target = Path(out_path)
    if target.exists() and target.is_dir():
        return target / _default_export_filename(dataset_name, fmt)
    return target


def cmd_export(args):
    dataset_arg = getattr(args, "dataset", None)
    if dataset_arg:
        d = _resolve_dataset_for_read(args.root, str(dataset_arg))
    else:
        ds_name = _resolve_dataset_name_interactive(args.root, dataset_arg, False)
        if not ds_name:
            return
        d = Dataset(args.root, ds_name)
    fmt = str(args.fmt or "").strip().lower()
    out_target = _resolve_export_target(Path(args.out), dataset_name=d.name, fmt=fmt)
    cols = [c.strip() for c in args.columns.split(",") if c.strip()] if args.columns else None
    d.export(fmt, out_target, columns=cols, include_deleted=bool(getattr(args, "include_deleted", False)))
    print(f"Wrote {out_target}")


def _collect_ids(ids: list[str] | None, id_file: Path | None) -> list[str]:
    out: list[str] = []
    if ids:
        for v in ids:
            out.extend([s.strip() for s in str(v).split(",") if s.strip()])
    if id_file:
        text = Path(id_file).read_text(encoding="utf-8")
        out.extend([line.strip() for line in text.splitlines() if line.strip()])
    if not out:
        raise SequencesError("Provide at least one id via --id or --id-file.")
    return out


def _collect_list(vals: list[str] | None) -> list[str]:
    out: list[str] = []
    if vals:
        for v in vals:
            out.extend([s.strip() for s in str(v).split(",") if s.strip()])
    return out


def cmd_delete(args):
    state_commands.cmd_delete(
        args,
        collect_ids=_collect_ids,
    )


def cmd_restore(args):
    state_commands.cmd_restore(
        args,
        collect_ids=_collect_ids,
    )


def cmd_state_set(args):
    state_commands.cmd_state_set(
        args,
        collect_ids=_collect_ids,
        collect_list=_collect_list,
    )


def cmd_state_clear(args):
    state_commands.cmd_state_clear(
        args,
        collect_ids=_collect_ids,
    )


def cmd_state_get(args):
    state_commands.cmd_state_get(
        args,
        collect_ids=_collect_ids,
        resolve_output_format=_resolve_output_format,
        print_json=_print_json,
        output_version=USR_OUTPUT_VERSION,
    )


def cmd_materialize(args):
    ds_name = _resolve_dataset_name_interactive(args.root, getattr(args, "dataset", None), False)
    if not ds_name:
        return
    d = Dataset(args.root, ds_name)
    ns_filter = [s.strip() for s in (args.namespaces or "").split(",") if s.strip()]
    overlays = list_overlays(d.dir)
    if ns_filter:
        ns_set = set(ns_filter)
        overlays = [p for p in overlays if (overlay_metadata(p).get("namespace") or p.stem) in ns_set]
    else:
        overlays = [p for p in overlays if (overlay_metadata(p).get("namespace") or p.stem) != "usr"]
    if not overlays and not getattr(args, "drop_deleted", False):
        print("No overlays found to materialize.")
        return
    if getattr(args, "drop_overlays", False) and getattr(args, "archive_overlays", False):
        raise SequencesError("Choose either --drop-overlays or --archive-overlays (not both).")
    if _is_interactive() and not getattr(args, "yes", False):
        overlay_names = []
        for path in overlays:
            meta = overlay_metadata(path)
            ns = meta.get("namespace") or path.stem
            overlay_names.append(ns)
        ns_list = ", ".join(sorted(set(overlay_names)))
        print("WARNING: materialize will rewrite records.parquet by merging overlays into the base table.")
        print(f"Dataset: {d.name}")
        print(f"Overlays: {ns_list or '(none)'}")
        if getattr(args, "drop_overlays", False):
            print("Overlays: will be deleted after materialize.")
        elif getattr(args, "archive_overlays", False):
            print("Overlays: will be archived after materialize.")
        else:
            print("Overlays: will be kept after materialize.")
        if getattr(args, "drop_deleted", False):
            print("Deleted rows will be dropped from the base table.")
        if not getattr(args, "snapshot_before", False):
            print("Tip: pass --snapshot-before to save a snapshot first.")
        if not typer.confirm("Proceed?", default=False):
            raise UserAbort()
    if getattr(args, "snapshot_before", False):
        d.snapshot()
    with d.maintenance(reason="materialize"):
        d.materialize(
            namespaces=ns_filter or None,
            keep_overlays=not bool(getattr(args, "drop_overlays", False)),
            archive_overlays=bool(getattr(args, "archive_overlays", False)),
            drop_deleted=bool(getattr(args, "drop_deleted", False)),
        )
    print(f"Materialized {len(overlays)} overlay(s) into {d.records_path}")
    d.append_meta_note("Materialized overlays", f"usr materialize {ds_name}")


def cmd_snapshot(args):
    ds_name = _resolve_dataset_name_interactive(args.root, getattr(args, "dataset", None), False)
    if not ds_name:
        return
    d = Dataset(args.root, ds_name)
    d.snapshot()
    print(f"Snapshot saved under {d.snapshot_dir}")
    d.append_meta_note("Snapshot saved", f"usr snapshot {ds_name}")


def cmd_convert_legacy(args):
    from .convert_legacy import convert_legacy, profile_60bp_dual_promoter

    in_paths = [_resolve_path_anywhere(p) for p in args.paths]

    stats = convert_legacy(
        dataset_root=args.root,
        dataset_name=args.dataset,
        pt_paths=in_paths,
        profile=profile_60bp_dual_promoter(),  # current supported profile
        expected_length=args.expected_length,
        plan_override=args.plan,
        force=bool(args.force),
    )

    msg = f"Converted {stats.rows} row(s) from {stats.files} file(s) into dataset '{args.dataset}'."
    if stats.skipped_bad_len:
        msg += f" Skipped (length≠expected): {stats.skipped_bad_len}."
    print(msg)


# ---------- make-mock ----------
def cmd_make_mock(args):
    spec = MockSpec(
        n=int(args.n),
        length=int(args.length),
        x_dim=int(args.x_dim),
        y_dim=int(args.y_dim),
        seed=int(args.seed),
        namespace=str(args.namespace),
        csv_path=_resolve_path_anywhere(args.from_csv) if args.from_csv else None,
    )
    created = create_mock_dataset(args.root, args.dataset, spec, force=bool(args.force))
    print(
        f"Created mock dataset '{args.dataset}' with {created} rows, "
        f"{spec.namespace}__x_representation[{spec.x_dim}] and {spec.namespace}__label_vec8[{spec.y_dim}]"
        + (" (from CSV)" if args.from_csv else " (random sequences)")
    )
    src = f"--from-csv {spec.csv_path}" if spec.csv_path else f"--length {spec.length}"
    cmd = f"usr make-mock {args.dataset} --n {spec.n} {src} --namespace {spec.namespace} --x-dim {spec.x_dim} --y-dim {spec.y_dim}"  # noqa
    Dataset(args.root, args.dataset).append_meta_note("Created mock dataset", cmd)


# ---------- add-demo-cols ----------
def cmd_add_demo(args):
    n = add_demo_columns(
        args.root,
        args.dataset,
        x_dim=int(args.x_dim),
        y_dim=int(args.y_dim),
        seed=int(args.seed),
        namespace=str(args.namespace),
        allow_overwrite=bool(args.allow_overwrite),
    )
    print(
        f"Added demo columns to {n} rows in '{args.dataset}' "
        f"({args.namespace}__x_representation[{args.x_dim}], {args.namespace}__label_vec8[{args.y_dim}])."
    )
    cmd = f"usr add-demo-cols {args.dataset} --x-dim {args.x_dim} --y-dim {args.y_dim} --namespace {args.namespace}"
    Dataset(args.root, args.dataset).append_meta_note("Added demo columns", cmd)


# ---------- MERGE DATASETS ----------
def cmd_merge_datasets(args):
    columns = str(getattr(args, "columns", "") or "")
    cols_subset = [c.strip() for c in columns.split(",") if c.strip()] if columns else None
    require_same = bool(getattr(args, "require_same", False))
    mode = MergeColumnsMode.REQUIRE_SAME if require_same else MergeColumnsMode.UNION
    dup_policy = str(getattr(args, "dup_policy", "error") or "error")
    policy = resolve_merge_policy(dup_policy)
    coerce_overlap = str(getattr(args, "coerce_overlap", "none") or "none")
    if coerce_overlap not in {"none", "to-dest"}:
        raise SequencesError("--coerce-overlap must be one of: none, to-dest")
    dry_run = bool(getattr(args, "dry_run", False))
    assume_yes = bool(getattr(args, "yes", False))
    note = str(getattr(args, "note", "") or "")
    if hasattr(args, "avoid_casefold_dups"):
        avoid_casefold_dups = bool(getattr(args, "avoid_casefold_dups", True))
    else:
        avoid_casefold_dups = not bool(getattr(args, "no_avoid_casefold_dups", False))

    ds_dest = Dataset(args.root, args.dest)
    with ds_dest.maintenance(reason="merge"):
        preview = merge_usr_to_usr(
            root=args.root,
            dest=args.dest,
            src=args.src,
            columns_mode=mode,
            duplicate_policy=policy,
            columns_subset=cols_subset,
            dry_run=dry_run,
            assume_yes=assume_yes,
            note=note,
            overlap_coercion=("to-dest" if coerce_overlap == "to-dest" else "none"),
            avoid_casefold_dups=avoid_casefold_dups,
        )

    action = "DRY-RUN" if dry_run else "MERGED"
    print(
        f"[{action}] dest='{args.dest}'  src='{args.src}'  "
        f"rows_src={preview.src_rows}  "
        f"duplicates_total={preview.duplicates_total}  "
        f"duplicates_skipped={preview.duplicates_skipped}  "
        f"duplicates_replaced={preview.duplicates_replaced}  "
        f"rows_added={preview.new_rows}  "
        f"dest_rows: {preview.dest_rows_before} → {preview.dest_rows_after}  "
        f"columns(total={preview.columns_total}, overlap={preview.overlapping_columns})  "
        f"dup_policy={preview.duplicate_policy.value}"
    )
    if not dry_run:
        cmd = (
            f"usr merge-datasets --dest {args.dest} --src {args.src} "
            f"{'--require-same-columns' if require_same else '--union-columns'} "
            f"--if-duplicate {dup_policy}"
        )
        ds_dest.append_meta_note(
            f"Merged from '{args.src}' → '{args.dest}' (added {preview.new_rows} rows; dup_policy={preview.duplicate_policy.value})",  # noqa
            cmd,
        )


# ---------- remotes commands ----------
def cmd_remotes_list(args):
    remotes_commands.cmd_remotes_list(args)


def cmd_remotes_show(args):
    remotes_commands.cmd_remotes_show(args)


def cmd_remotes_add(args):
    remotes_commands.cmd_remotes_add(args)


def cmd_remotes_wizard(args):
    remotes_commands.cmd_remotes_wizard(args)


def cmd_remotes_doctor(args):
    remotes_commands.shutil = shutil
    remotes_commands.SSHRemote = SSHRemote
    remotes_commands.cmd_remotes_doctor(args)


# ---------- namespace registry ----------
def cmd_namespace_list(args):
    entries = load_registry(args.root, required=True)
    if not entries:
        print("(no namespaces registered)")
        return
    for name, entry in sorted(entries.items()):
        cols = ", ".join(c.name for c in entry.columns)
        print(f"{name}: {cols}")


def cmd_namespace_show(args):
    entries = load_registry(args.root, required=True)
    if args.name not in entries:
        raise SystemExit(f"Namespace '{args.name}' not registered.")
    entry = entries[args.name]
    print(f"name: {entry.namespace}")
    print(f"owner: {entry.owner or ''}")
    print(f"description: {entry.description or ''}")
    print("columns:")
    for col in entry.columns:
        print(f"  - {col.name}: {col.type}")


def cmd_namespace_register(args):
    cols = parse_columns_spec(args.columns, namespace=args.namespace)
    path = register_namespace(
        args.root,
        namespace=args.namespace,
        columns=cols,
        owner=args.owner,
        description=args.description,
        overwrite=bool(args.overwrite),
    )
    print(f"Registered namespace '{args.namespace}' in {path}.")


# ---------- diff/pull/push ----------
def cmd_diff(args):
    sync_commands.cmd_diff(
        args,
        resolve_output_format=_resolve_output_format,
        print_json=_print_json,
        output_version=USR_OUTPUT_VERSION,
    )


def cmd_pull(args):
    sync_commands.cmd_pull(args)


def cmd_push(args):
    sync_commands.cmd_push(args)


def cmd_dedupe_sequences(args):
    d = Dataset(args.root, args.dataset)
    with d.maintenance(reason="dedupe"):
        stats = d.dedupe(
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
        ans = input("Proceed with destructive de-duplication? [y/N]: ").strip().lower()
        if ans not in {"y", "yes"}:
            print("Aborted.")
            return
    with d.maintenance(reason="dedupe"):
        stats = d.dedupe(
            key=str(args.key),
            keep=str(args.keep),
            batch_size=int(args.batch_size),
            dry_run=False,
        )
    rows_after = stats.rows_total - stats.rows_dropped
    print(f"[dedupe] dropped {stats.rows_dropped} row(s); dataset now has {rows_after} rows.")


# ---------- Typer CLI (library-first adapter) ----------
app = typer.Typer(add_completion=True, no_args_is_help=True, help="USR datasets CLI (Typer).")
remotes_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Manage SSH remotes.")
legacy_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Legacy dataset utilities.")
maintenance_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Dataset maintenance utilities.")
densegen_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Densegen-specific utilities.")
dev_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Developer utilities (unstable).")
namespace_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Manage namespace registry.")
events_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Inspect dataset events.")
state_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Record state utilities.")

app.add_typer(remotes_app, name="remotes")
app.add_typer(legacy_app, name="legacy")
app.add_typer(maintenance_app, name="maintenance")
app.add_typer(densegen_app, name="densegen")
app.add_typer(namespace_app, name="namespace")
app.add_typer(events_app, name="events")
app.add_typer(state_app, name="state")
if os.getenv("USR_SHOW_DEV_COMMANDS") == "1":
    app.add_typer(dev_app, name="dev")


def _ctx_args(ctx: typer.Context, **kwargs) -> NS:
    base = {"root": ctx.obj["root"], "rich": ctx.obj["rich"]}
    base.update(kwargs)
    return NS(**base)


@app.callback()
def _root(
    ctx: typer.Context,
    root: Path = typer.Option(
        (_pkg_usr_root() / "datasets").resolve(),
        "--root",
        help="Datasets root folder",
        readable=True,
        exists=True,
        dir_okay=True,
        file_okay=False,
        path_type=Path,
    ),
    rich: bool = typer.Option(True, "--rich/--no-rich", help="Use Rich formatting for supported commands"),
) -> None:
    try:
        _assert_supported_root(root)
    except SequencesError as exc:
        raise typer.BadParameter(str(exc), param_hint="--root") from exc
    ctx.obj = {"root": root, "rich": rich}


@app.command("ls")
def cli_ls(
    ctx: typer.Context,
    format: str = typer.Option("auto", "--format", help="Output format: auto|rich|plain|json"),
) -> None:
    cmd_ls(_ctx_args(ctx, format=format))


@app.command("init")
def cli_init(
    ctx: typer.Context,
    dataset: str = typer.Argument(...),
    source: str = typer.Option("", "--source"),
    notes: str = typer.Option("", "--notes"),
) -> None:
    cmd_init(_ctx_args(ctx, dataset=dataset, source=source, notes=notes))


@app.command("import")
def cli_import(
    ctx: typer.Context,
    dataset: str = typer.Argument(...),
    source_format: str = typer.Option(..., "--from", help="Source format", case_sensitive=False),
    path: Path = typer.Option(..., "--path", exists=True, readable=True, path_type=Path),
    bio_type: str = typer.Option("dna", "--bio-type"),
    alphabet: str = typer.Option("dna_4", "--alphabet"),
) -> None:
    cmd_import(
        _ctx_args(
            ctx,
            dataset=dataset,
            source_format=source_format,
            path=path,
            bio_type=bio_type,
            alphabet=alphabet,
        )
    )


@app.command("attach")
def cli_attach(
    ctx: typer.Context,
    dataset: str = typer.Argument(...),
    path: Path = typer.Option(..., "--path", exists=True, readable=True, path_type=Path),
    namespace: str = typer.Option(..., "--namespace"),
    key: str = typer.Option(..., "--key"),
    key_col: str = typer.Option("", "--key-col"),
    columns: str = typer.Option("", "--columns"),
    allow_overwrite: bool = typer.Option(False, "--allow-overwrite"),
    allow_missing: bool = typer.Option(False, "--allow-missing"),
    parse_json: bool = typer.Option(True, "--parse-json/--no-parse-json"),
    backend: str = typer.Option("pyarrow", "--backend"),
    note: str = typer.Option("", "--note"),
) -> None:
    cmd_attach(
        _ctx_args(
            ctx,
            dataset=dataset,
            path=path,
            namespace=namespace,
            key=key,
            key_col=key_col,
            columns=columns,
            allow_overwrite=allow_overwrite,
            allow_missing=allow_missing,
            parse_json=parse_json,
            backend=backend,
            note=note,
        )
    )


@app.command("info")
def cli_info(
    ctx: typer.Context,
    dataset: str = typer.Argument(None),
    format: str = typer.Option("auto", "--format", help="Output format: auto|rich|plain|json"),
) -> None:
    cmd_info(_ctx_args(ctx, dataset=dataset, format=format))


@app.command("schema")
def cli_schema(
    ctx: typer.Context,
    dataset: str = typer.Argument(None),
    tree: bool = typer.Option(False, "--tree"),
    format: str = typer.Option("auto", "--format", help="Output format: auto|rich|plain|json"),
) -> None:
    cmd_schema(_ctx_args(ctx, dataset=dataset, tree=tree, format=format))


@app.command("head")
def cli_head(
    ctx: typer.Context,
    target: str = typer.Argument(".", help="Dataset name or file/directory path"),
    n: int = typer.Option(10, "-n"),
    columns: str = typer.Option("", "--columns"),
    include_deleted: bool = typer.Option(False, "--include-deleted"),
    raw: bool = typer.Option(False, "--raw"),
    max_colwidth: int = typer.Option(80, "--max-colwidth"),
    max_list_items: int = typer.Option(6, "--max-list-items"),
    precision: int = typer.Option(4, "--precision"),
) -> None:
    cmd_head(
        _ctx_args(
            ctx,
            target=target,
            n=n,
            columns=columns,
            include_deleted=include_deleted,
            raw=raw,
            max_colwidth=max_colwidth,
            max_list_items=max_list_items,
            precision=precision,
        )
    )


@app.command("cols")
def cli_cols(
    ctx: typer.Context,
    target: str = typer.Argument(".", help="Dataset name or file/directory path"),
    glob: str | None = typer.Option(None, "--glob"),
) -> None:
    cmd_cols(_ctx_args(ctx, target=target, glob=glob))


@app.command("describe")
def cli_describe(
    ctx: typer.Context,
    dataset: str = typer.Argument(...),
    columns: str = typer.Option("", "--columns"),
    sample: int = typer.Option(1024, "--sample"),
    include_deleted: bool = typer.Option(False, "--include-deleted"),
    max_colwidth: int = typer.Option(80, "--max-colwidth"),
    max_list_items: int = typer.Option(6, "--max-list-items"),
    precision: int = typer.Option(4, "--precision"),
) -> None:
    cmd_describe(
        _ctx_args(
            ctx,
            dataset=dataset,
            columns=columns,
            sample=sample,
            include_deleted=include_deleted,
            max_colwidth=max_colwidth,
            max_list_items=max_list_items,
            precision=precision,
        )
    )


@app.command("cell")
def cli_cell(
    ctx: typer.Context,
    target: str = typer.Argument(".", help="Dataset name or file/directory path"),
    row: int = typer.Option(0, "--row"),
    col: str = typer.Option("", "--col"),
    glob: str | None = typer.Option(None, "--glob"),
) -> None:
    cmd_cell(_ctx_args(ctx, target=target, row=row, col=col, glob=glob))


@app.command("validate")
def cli_validate(
    ctx: typer.Context,
    dataset: str = typer.Argument(None),
    strict: bool = typer.Option(False, "--strict"),
    registry_mode: str = typer.Option(
        "current",
        "--registry-mode",
        help="Registry mode: current|frozen|either",
    ),
) -> None:
    cmd_validate(_ctx_args(ctx, dataset=dataset, strict=strict, registry_mode=registry_mode))


@events_app.command("tail")
def cli_events_tail(
    ctx: typer.Context,
    dataset: str = typer.Argument(None),
    format: str = typer.Option("json", "--format", help="Output format: json|raw"),
    n: int = typer.Option(0, "--n", help="Show only the last N events (0 = all)."),
    follow: bool = typer.Option(False, "--follow", help="Follow the events log for new entries."),
) -> None:
    cmd_events_tail(_ctx_args(ctx, dataset=dataset, format=format, n=n, follow=follow))


@app.command("get")
def cli_get(
    ctx: typer.Context,
    dataset: str = typer.Argument(...),
    record_id: str = typer.Option(..., "--id"),
    columns: str = typer.Option("", "--columns"),
    include_deleted: bool = typer.Option(False, "--include-deleted"),
) -> None:
    cmd_get(_ctx_args(ctx, dataset=dataset, id=record_id, columns=columns, include_deleted=include_deleted))


@app.command("grep")
def cli_grep(
    ctx: typer.Context,
    dataset: str = typer.Argument(...),
    pattern: str = typer.Option(..., "--pattern"),
    limit: int = typer.Option(20, "--limit"),
    batch_size: int = typer.Option(65536, "--batch-size"),
    include_deleted: bool = typer.Option(False, "--include-deleted"),
) -> None:
    cmd_grep(
        _ctx_args(
            ctx,
            dataset=dataset,
            pattern=pattern,
            limit=limit,
            batch_size=batch_size,
            include_deleted=include_deleted,
        )
    )


@app.command("export")
def cli_export(
    ctx: typer.Context,
    dataset: str = typer.Argument(None),
    fmt: str = typer.Option(..., "--fmt", help="Export format: csv|jsonl|parquet."),
    out: Path = typer.Option(
        ...,
        "--out",
        path_type=Path,
        help="Output file path, or an existing directory for auto-named export output.",
    ),
    columns: str = typer.Option("", "--columns"),
    include_deleted: bool = typer.Option(False, "--include-deleted"),
) -> None:
    cmd_export(_ctx_args(ctx, dataset=dataset, fmt=fmt, out=out, columns=columns, include_deleted=include_deleted))


@app.command("delete")
def cli_delete(
    ctx: typer.Context,
    dataset: str = typer.Argument(None),
    id: list[str] = typer.Option(None, "--id"),
    id_file: Path | None = typer.Option(None, "--id-file"),
    reason: str = typer.Option("", "--reason"),
    allow_missing: bool = typer.Option(False, "--allow-missing"),
) -> None:
    cmd_delete(
        _ctx_args(
            ctx,
            dataset=dataset,
            id=id,
            id_file=id_file,
            reason=reason,
            allow_missing=allow_missing,
        )
    )


@app.command("restore")
def cli_restore(
    ctx: typer.Context,
    dataset: str = typer.Argument(None),
    id: list[str] = typer.Option(None, "--id"),
    id_file: Path | None = typer.Option(None, "--id-file"),
    allow_missing: bool = typer.Option(False, "--allow-missing"),
) -> None:
    cmd_restore(
        _ctx_args(
            ctx,
            dataset=dataset,
            id=id,
            id_file=id_file,
            allow_missing=allow_missing,
        )
    )


@state_app.command("set")
def cli_state_set(
    ctx: typer.Context,
    dataset: str = typer.Argument(None),
    id: list[str] = typer.Option(None, "--id"),
    id_file: Path | None = typer.Option(None, "--id-file"),
    masked: bool | None = typer.Option(None, "--masked/--unmasked"),
    qc_status: str = typer.Option("", "--qc-status"),
    split: str = typer.Option("", "--split"),
    supersedes: str = typer.Option("", "--supersedes"),
    lineage: list[str] = typer.Option(None, "--lineage"),
    allow_missing: bool = typer.Option(False, "--allow-missing"),
) -> None:
    cmd_state_set(
        _ctx_args(
            ctx,
            dataset=dataset,
            id=id,
            id_file=id_file,
            masked=masked,
            qc_status=qc_status,
            split=split,
            supersedes=supersedes,
            lineage=lineage,
            allow_missing=allow_missing,
        )
    )


@state_app.command("clear")
def cli_state_clear(
    ctx: typer.Context,
    dataset: str = typer.Argument(None),
    id: list[str] = typer.Option(None, "--id"),
    id_file: Path | None = typer.Option(None, "--id-file"),
    allow_missing: bool = typer.Option(False, "--allow-missing"),
) -> None:
    cmd_state_clear(
        _ctx_args(
            ctx,
            dataset=dataset,
            id=id,
            id_file=id_file,
            allow_missing=allow_missing,
        )
    )


@state_app.command("get")
def cli_state_get(
    ctx: typer.Context,
    dataset: str = typer.Argument(None),
    id: list[str] = typer.Option(None, "--id"),
    id_file: Path | None = typer.Option(None, "--id-file"),
    allow_missing: bool = typer.Option(False, "--allow-missing"),
    format: str = typer.Option("auto", "--format", help="Output format: auto|rich|plain|json"),
) -> None:
    cmd_state_get(
        _ctx_args(
            ctx,
            dataset=dataset,
            id=id,
            id_file=id_file,
            allow_missing=allow_missing,
            format=format,
        )
    )


@app.command("materialize")
def cli_materialize(
    ctx: typer.Context,
    dataset: str = typer.Argument(None),
    yes: bool = typer.Option(False, "--yes", "-y"),
    snapshot_before: bool = typer.Option(False, "--snapshot-before"),
    namespaces: str = typer.Option("", "--namespaces", help="Comma-separated overlay namespaces to materialize"),
    drop_overlays: bool = typer.Option(False, "--drop-overlays"),
    archive_overlays: bool = typer.Option(False, "--archive-overlays"),
    drop_deleted: bool = typer.Option(False, "--drop-deleted"),
) -> None:
    cmd_materialize(
        _ctx_args(
            ctx,
            dataset=dataset,
            yes=yes,
            snapshot_before=snapshot_before,
            namespaces=namespaces,
            drop_overlays=drop_overlays,
            archive_overlays=archive_overlays,
            drop_deleted=drop_deleted,
        )
    )


@app.command("snapshot")
def cli_snapshot(ctx: typer.Context, dataset: str = typer.Argument(None)) -> None:
    cmd_snapshot(_ctx_args(ctx, dataset=dataset))


@maintenance_app.command("dedupe")
def cli_dedupe_sequences(
    ctx: typer.Context,
    dataset: str = typer.Argument(...),
    key: str = typer.Option(..., "--key", help="Dedupe key: id|sequence|sequence_norm|sequence_ci"),
    keep: str = typer.Option("keep-first", "--keep", help="Which occurrence to keep: keep-first|keep-last"),
    batch_size: int = typer.Option(65536, "--batch-size", help="Parquet batch size for streaming dedupe"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    yes: bool = typer.Option(False, "--yes"),
) -> None:
    cmd_dedupe_sequences(
        _ctx_args(
            ctx,
            dataset=dataset,
            key=key,
            keep=keep,
            batch_size=batch_size,
            dry_run=dry_run,
            yes=yes,
        )
    )


@maintenance_app.command("registry-freeze")
def cli_registry_freeze(
    ctx: typer.Context,
    dataset: str = typer.Argument(...),
) -> None:
    cmd_registry_freeze(_ctx_args(ctx, dataset=dataset))


@maintenance_app.command("overlay-compact")
def cli_overlay_compact(
    ctx: typer.Context,
    dataset: str = typer.Argument(...),
    namespace: str = typer.Option(..., "--namespace"),
) -> None:
    cmd_overlay_compact(_ctx_args(ctx, dataset=dataset, namespace=namespace))


@densegen_app.command("repair")
def cli_repair_densegen(
    ctx: typer.Context,
    dataset: str = typer.Argument(None),
    min_tfbs_len: int = typer.Option(6, "--min-tfbs-len"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    yes: bool = typer.Option(False, "--yes"),
    dedupe: str = typer.Option("off", "--dedupe"),
    drop_missing_used_tfbs: bool = typer.Option(False, "--drop-missing-used-tfbs"),
    drop_single_tf: bool = typer.Option(False, "--drop-single-tf"),
    drop_id_seq_only: bool = typer.Option(False, "--drop-id-seq-only"),
    filter_single_tf: bool = typer.Option(False, "--filter-single-tf"),
) -> None:
    cmd_repair_densegen(
        _ctx_args(
            ctx,
            dataset=dataset,
            min_tfbs_len=min_tfbs_len,
            dry_run=dry_run,
            yes=yes,
            dedupe=dedupe,
            drop_missing_used_tfbs=drop_missing_used_tfbs,
            drop_single_tf=drop_single_tf,
            drop_id_seq_only=drop_id_seq_only,
            filter_single_tf=filter_single_tf,
        )
    )


@dev_app.command("make-mock")
def cli_make_mock(
    ctx: typer.Context,
    dataset: str = typer.Argument(...),
    n: int = typer.Option(100, "--n"),
    length: int = typer.Option(60, "--length"),
    x_dim: int = typer.Option(512, "--x-dim"),
    y_dim: int = typer.Option(8, "--y-dim"),
    seed: int = typer.Option(7, "--seed"),
    namespace: str = typer.Option("demo", "--namespace"),
    from_csv: str = typer.Option("", "--from-csv"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    cmd_make_mock(
        _ctx_args(
            ctx,
            dataset=dataset,
            n=n,
            length=length,
            x_dim=x_dim,
            y_dim=y_dim,
            seed=seed,
            namespace=namespace,
            from_csv=from_csv,
            force=force,
        )
    )


@dev_app.command("add-demo-cols")
def cli_add_demo_cols(
    ctx: typer.Context,
    dataset: str = typer.Argument(...),
    x_dim: int = typer.Option(512, "--x-dim"),
    y_dim: int = typer.Option(8, "--y-dim"),
    seed: int = typer.Option(7, "--seed"),
    namespace: str = typer.Option("demo", "--namespace"),
    allow_overwrite: bool = typer.Option(False, "--allow-overwrite"),
) -> None:
    cmd_add_demo(
        _ctx_args(
            ctx,
            dataset=dataset,
            x_dim=x_dim,
            y_dim=y_dim,
            seed=seed,
            namespace=namespace,
            allow_overwrite=allow_overwrite,
        )
    )


@remotes_app.command("list")
def cli_remotes_list(ctx: typer.Context) -> None:
    cmd_remotes_list(_ctx_args(ctx))


@remotes_app.command("show")
def cli_remotes_show(ctx: typer.Context, name: str = typer.Argument(...)) -> None:
    cmd_remotes_show(_ctx_args(ctx, name=name))


@remotes_app.command("add")
def cli_remotes_add(
    ctx: typer.Context,
    name: str = typer.Argument(...),
    type: str = typer.Option("ssh", "--type"),
    host: str = typer.Option(..., "--host"),
    user: str = typer.Option(..., "--user"),
    base_dir: str = typer.Option(..., "--base-dir"),
    ssh_key_env: str | None = typer.Option(None, "--ssh-key-env"),
) -> None:
    cmd_remotes_add(
        _ctx_args(
            ctx,
            name=name,
            type=type,
            host=host,
            user=user,
            base_dir=base_dir,
            ssh_key_env=ssh_key_env,
        )
    )


@remotes_app.command("wizard")
def cli_remotes_wizard(
    ctx: typer.Context,
    preset: str = typer.Option("bu-scc", "--preset", help="Wizard preset: bu-scc."),
    name: str = typer.Option("bu-scc", "--name", help="Remote config name."),
    user: str = typer.Option(..., "--user", help="Remote SSH username."),
    base_dir: str = typer.Option(..., "--base-dir", help="Remote dataset root path."),
    host: str = typer.Option(
        "",
        "--host",
        help="Remote host override. Defaults to scc1.bu.edu, or scc-globus.bu.edu with --transfer-node.",
    ),
    transfer_node: bool = typer.Option(
        False,
        "--transfer-node",
        help="Use BU SCC transfer host default (scc-globus.bu.edu).",
    ),
    ssh_key_env: str | None = typer.Option(None, "--ssh-key-env"),
) -> None:
    cmd_remotes_wizard(
        _ctx_args(
            ctx,
            preset=preset,
            name=name,
            user=user,
            base_dir=base_dir,
            host=host,
            transfer_node=transfer_node,
            ssh_key_env=ssh_key_env,
        )
    )


@remotes_app.command("doctor")
def cli_remotes_doctor(
    ctx: typer.Context,
    remote: str = typer.Option(..., "--remote", help="Configured remote name."),
    check_base_dir: bool = typer.Option(True, "--check-base-dir/--no-check-base-dir"),
) -> None:
    cmd_remotes_doctor(_ctx_args(ctx, remote=remote, check_base_dir=check_base_dir))


@namespace_app.command("list")
def cli_namespace_list(ctx: typer.Context) -> None:
    cmd_namespace_list(_ctx_args(ctx))


@namespace_app.command("show")
def cli_namespace_show(ctx: typer.Context, name: str = typer.Argument(...)) -> None:
    cmd_namespace_show(_ctx_args(ctx, name=name))


@namespace_app.command("register")
def cli_namespace_register(
    ctx: typer.Context,
    namespace: str = typer.Argument(...),
    columns: str = typer.Option(..., "--columns", help="Comma-separated name:type list"),
    owner: str = typer.Option("", "--owner"),
    description: str = typer.Option("", "--description"),
    overwrite: bool = typer.Option(False, "--overwrite"),
) -> None:
    cmd_namespace_register(
        _ctx_args(
            ctx,
            namespace=namespace,
            columns=columns,
            owner=owner or None,
            description=description or None,
            overwrite=overwrite,
        )
    )


@legacy_app.command("convert")
def cli_convert_legacy(
    ctx: typer.Context,
    dataset: str = typer.Argument(...),
    paths: list[Path] = typer.Argument(...),
    expected_length: int | None = typer.Option(None, "--expected-length"),
    plan: str | None = typer.Option(None, "--plan"),
    force: bool = typer.Option(False, "--force"),
    profile_60bp: bool = typer.Option(True, "--profile-60bp/--no-profile-60bp"),
) -> None:
    cmd_convert_legacy(
        _ctx_args(
            ctx,
            dataset=dataset,
            paths=paths,
            expected_length=expected_length,
            plan=plan,
            force=force,
            profile_60bp=profile_60bp,
        )
    )


@maintenance_app.command("merge")
def cli_merge_datasets(
    ctx: typer.Context,
    dest: str = typer.Option(..., "--dest"),
    src: str = typer.Option(..., "--src"),
    require_same_columns: bool = typer.Option(False, "--require-same-columns"),
    union_columns: bool = typer.Option(False, "--union-columns"),
    dup_policy: str = typer.Option("error", "--if-duplicate"),
    coerce_overlap: str = typer.Option("none", "--coerce-overlap"),
    no_avoid_casefold_dups: bool = typer.Option(False, "--no-avoid-casefold-dups"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    cmd_merge_datasets(
        _ctx_args(
            ctx,
            dest=dest,
            src=src,
            require_same=require_same_columns,
            union_columns=union_columns,
            dup_policy=dup_policy,
            coerce_overlap=coerce_overlap,
            no_avoid_casefold_dups=no_avoid_casefold_dups,
            dry_run=dry_run,
        )
    )


def _sync_args(
    ctx: typer.Context,
    dataset: str,
    remote: str,
    primary_only: bool,
    skip_snapshots: bool,
    dry_run: bool,
    yes: bool,
    verify: str,
    format: str | None,
    repo_root: str | None,
    remote_path: str | None,
) -> NS:
    return _ctx_args(
        ctx,
        dataset=dataset,
        remote=remote,
        primary_only=primary_only,
        skip_snapshots=skip_snapshots,
        dry_run=dry_run,
        yes=yes,
        verify=verify,
        format=format,
        repo_root=repo_root,
        remote_path=remote_path,
    )


@app.command("diff")
def cli_diff(
    ctx: typer.Context,
    dataset: str = typer.Argument(...),
    remote: str = typer.Argument(...),
    primary_only: bool = typer.Option(False, "--primary-only"),
    skip_snapshots: bool = typer.Option(False, "--skip-snapshots"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    yes: bool = typer.Option(False, "--yes"),
    verify: str = typer.Option("auto", "--verify", help="Verification mode: auto|hash|size|parquet"),
    format: str = typer.Option("auto", "--format", help="Output format: auto|rich|plain|json"),
    repo_root: str | None = typer.Option(None, "--repo-root"),
    remote_path: str | None = typer.Option(None, "--remote-path"),
) -> None:
    cmd_diff(
        _sync_args(
            ctx,
            dataset,
            remote,
            primary_only,
            skip_snapshots,
            dry_run,
            yes,
            verify,
            format,
            repo_root,
            remote_path,
        )
    )


@app.command("status")
def cli_status(
    ctx: typer.Context,
    dataset: str = typer.Argument(...),
    remote: str = typer.Argument(...),
    primary_only: bool = typer.Option(False, "--primary-only"),
    skip_snapshots: bool = typer.Option(False, "--skip-snapshots"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    yes: bool = typer.Option(False, "--yes"),
    verify: str = typer.Option("auto", "--verify", help="Verification mode: auto|hash|size|parquet"),
    repo_root: str | None = typer.Option(None, "--repo-root"),
    remote_path: str | None = typer.Option(None, "--remote-path"),
) -> None:
    cmd_diff(
        _sync_args(
            ctx,
            dataset,
            remote,
            primary_only,
            skip_snapshots,
            dry_run,
            yes,
            verify,
            None,
            repo_root,
            remote_path,
        )
    )


@app.command("pull")
def cli_pull(
    ctx: typer.Context,
    dataset: str = typer.Argument(...),
    remote: str = typer.Argument(...),
    primary_only: bool = typer.Option(False, "--primary-only"),
    skip_snapshots: bool = typer.Option(False, "--skip-snapshots"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    yes: bool = typer.Option(False, "--yes"),
    verify: str = typer.Option("auto", "--verify", help="Verification mode: auto|hash|size|parquet"),
    repo_root: str | None = typer.Option(None, "--repo-root"),
    remote_path: str | None = typer.Option(None, "--remote-path"),
) -> None:
    cmd_pull(
        _sync_args(
            ctx,
            dataset,
            remote,
            primary_only,
            skip_snapshots,
            dry_run,
            yes,
            verify,
            None,
            repo_root,
            remote_path,
        )
    )


@app.command("push")
def cli_push(
    ctx: typer.Context,
    dataset: str = typer.Argument(...),
    remote: str = typer.Argument(...),
    primary_only: bool = typer.Option(False, "--primary-only"),
    skip_snapshots: bool = typer.Option(False, "--skip-snapshots"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    yes: bool = typer.Option(False, "--yes"),
    verify: str = typer.Option("auto", "--verify", help="Verification mode: auto|hash|size|parquet"),
    repo_root: str | None = typer.Option(None, "--repo-root"),
    remote_path: str | None = typer.Option(None, "--remote-path"),
) -> None:
    cmd_push(
        _sync_args(
            ctx,
            dataset,
            remote,
            primary_only,
            skip_snapshots,
            dry_run,
            yes,
            verify,
            None,
            repo_root,
            remote_path,
        )
    )


def main() -> None:
    from .stderr_filter import maybe_install_pyarrow_sysctl_filter

    maybe_install_pyarrow_sysctl_filter()
    try:
        app()
    except UserAbort:
        raise SystemExit(130)
    except SequencesError as e:
        _print_user_error(e)
        raise SystemExit(2)
    except FileExistsError as e:
        print(f"ERROR: {e}")
        raise SystemExit(3)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        raise SystemExit(4)


if __name__ == "__main__":
    main()
