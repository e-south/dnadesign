"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/usr/src/cli.py

USR CLI: thin wrapper around the Dataset API.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .config import SSHRemoteConfig, get_remote, load_all, save_remote
from .convert_legacy import convert_legacy, profile_60bp_dual_promoter
from .dataset import Dataset
from .errors import DuplicateIDError, SequencesError, UserAbort
from .io import append_event
from .merge_datasets import (
    MergeColumnsMode,
    MergePolicy,
    merge_usr_to_usr,
)
from .mock import MockSpec, add_demo_columns, create_mock_dataset
from .pretty import PrettyOpts, fmt_value, profile_table, render_schema_tree
from .sync import (
    SyncOptions,
    execute_pull,
    execute_pull_file,
    execute_push,
    execute_push_file,
    plan_diff,
    plan_diff_file,
)
from .ui import (
    print_df_plain,
    render_diff_rich,
    render_schema_tree_rich,
    render_table_rich,
)
from .version import __version__


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


def _prompt_pick_parquet(cands: list[Path], use_rich: bool) -> Path | None:
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    # Build a compact table with mtime/rows/cols
    rows = []
    for idx, p in enumerate(cands, start=1):
        try:
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
        except Exception:
            rows.append(
                {
                    "#": idx,
                    "file": p.name,
                    "rows": "?",
                    "cols": "?",
                    "size": _human_size(None),
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
    except Exception:
        pass
    print(f"Invalid selection. Using newest: {cands[0].name}")
    return cands[0]


def _select_parquet_target_interactive(
    path_like: Path,
    glob: str | None,
    use_rich: bool,
    confirm_if_inferred: bool = False,
) -> Path | None:
    p = Path(path_like)
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
        # When the caller didn't provide an explicit path, ask before auto-picking.
        if confirm_if_inferred and len(cands) == 1:
            msg = f"No explicit file path provided. Found '{cands[0].name}' in {p}. Proceed? [Y/n]"
            print(msg)
            ans = input("> ").strip().lower()
            if ans in {"n", "no"}:
                print("Aborted.")
                return None
        return _prompt_pick_parquet(cands, use_rich)
    # Not a file or directory → let caller handle dataset/other cases
    return None


# --------- dataset guessing (path-first) ----------
def _prompt_pick_dataset(root: Path, names: list[str], use_rich: bool) -> str | None:
    if not names:
        print(f"(no datasets under {root})")
        return None
    if len(names) == 1:
        return names[0]
    # preview rows/cols for each dataset
    rows = []
    for idx, name in enumerate(names, start=1):
        rp = root / name / "records.parquet"
        try:
            pf = pq.ParquetFile(str(rp))
            rows.append(
                {
                    "#": idx,
                    "dataset": name,
                    "rows": pf.metadata.num_rows,
                    "cols": pf.metadata.num_columns,
                }
            )
        except Exception:
            rows.append({"#": idx, "dataset": name, "rows": "?", "cols": "?"})
    df = pd.DataFrame(rows, columns=["#", "dataset", "rows", "cols"])
    msg = "Multiple datasets found. Choose one by number (Enter = first, q = abort):"
    if use_rich:
        render_table_rich(df, title="Pick a dataset", caption=str(root))
    else:
        print_df_plain(df)
        print(msg)
    sel = input("> ").strip().lower()
    if sel in {"q", "quit", "n"}:
        print("Aborted.")
        return None
    if not sel:
        return names[0]
    try:
        k = int(sel)
        if 1 <= k <= len(names):
            return names[k - 1]
    except Exception:
        pass
    print(f"Invalid selection. Using: {names[0]}")
    return names[0]


def _resolve_dataset_name_interactive(
    root: Path, dataset: str | None, use_rich: bool
) -> str | None:
    """
    If dataset is None, try to infer from CWD:
      - If CWD is <root>/<dataset>[/...], use that dataset
      - If CWD == <root>, prompt to pick a dataset
      - Else: if CWD is <root>/<dataset>/_snapshots, also resolve
    """
    if dataset:
        return dataset
    root = Path(root).resolve()
    cwd = Path.cwd().resolve()
    try:
        rel = cwd.relative_to(root)
        if rel.parts:
            # accept <dataset>, or deeper like <dataset>/_snapshots
            cand = rel.parts[0]
            if (root / cand / "records.parquet").exists():
                return cand
    except Exception:
        pass
    if cwd == root:
        from_names = list_datasets(root)
        return _prompt_pick_dataset(root, from_names, use_rich)
    # final attempt: parent of cwd within root that contains records.parquet
    p = cwd
    for _ in range(3):
        if (p / "records.parquet").exists() and p.parent == root:
            return p.name
        p = p.parent
    print(
        "Dataset not provided and could not be inferred from CWD. Run inside a dataset folder under --root or pass a dataset name."  # noqa
    )
    return None


# ---------------- path helpers: resolve paths relative to the installed package ----------------


def _pkg_usr_root() -> Path:
    """
    Return the installed dnadesign/usr package directory.
    This is stable no matter where the user runs 'usr' from.
    """
    # .../dnadesign/usr/src/cli.py -> parents[1] = .../dnadesign/usr
    return Path(__file__).resolve().parents[1]


def _resolve_path_anywhere(p: Path) -> Path:
    """
    Make file arguments robust:
      1) absolute path → as-is
      2) relative path existing under CWD → as-is
      3) otherwise, try to resolve relative to the installed dnadesign/usr package,
         including common repo-style prefixes like 'src/dnadesign/usr/...'
         or 'usr/...'.
    """
    p = Path(p)

    # 1) absolute
    if p.is_absolute() and p.exists():
        return p

    # 2) relative under CWD
    if p.exists():
        return p

    # 3) under installed package
    base = _pkg_usr_root()

    # direct join
    cand = base / p
    if cand.exists():
        return cand

    # repo-style prefix: src/dnadesign/usr/<...>
    parts = p.parts
    if "dnadesign" in parts and "usr" in parts:
        try:
            i = parts.index("dnadesign")
            if parts[i + 1] == "usr":
                sub = Path(*parts[i + 2 :])
                cand2 = base / sub
                if cand2.exists():
                    return cand2
        except Exception:
            pass

    # prefix starting with 'usr/...'
    if parts and parts[0] == "usr":
        cand3 = base / Path(*parts[1:])
        if cand3.exists():
            return cand3

    # give up (let caller raise if needed)
    return p


def main() -> None:
    # --- argparse with Rich-styled --help everywhere ---
    p = argparse.ArgumentParser(
        prog="usr",
        add_help=False,  # we will inject a Rich help action
        description="USR CLI (Parquet-backed datasets; includes SSH remotes sync).",
    )
    _add_rich_help(p)
    default_root = (_pkg_usr_root() / "datasets").resolve()
    p.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help=f"Datasets root folder (defaults to {default_root})",
    )
    # Rich toggle (global). When None, default to plain text.
    p.add_argument(
        "--rich",
        dest="rich",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Rich formatting for supported commands (default: on).",
    )

    sp = p.add_subparsers(dest="cmd", required=True)

    # ---------------- core dataset commands ----------------
    sp_ls = sp.add_parser(
        "ls",
        add_help=False,
        help="List datasets under root",
        description="List datasets under root",
    )
    _add_rich_help(sp_ls)
    sp_ls.set_defaults(func=cmd_ls)

    sp_init = sp.add_parser(
        "init",
        add_help=False,
        help="Create an empty dataset",
        description="Create an empty dataset",
    )
    _add_rich_help(sp_init)
    sp_init.add_argument("dataset")
    sp_init.add_argument("--source", default="")
    sp_init.add_argument("--notes", default="")
    sp_init.set_defaults(func=cmd_init)

    sp_imp = sp.add_parser(
        "import",
        add_help=False,
        help="Import sequences into a dataset",
        description="Import sequences into a dataset",
    )
    _add_rich_help(sp_imp)
    sp_imp.add_argument("dataset")
    sp_imp.add_argument(
        "--from", dest="source_format", choices=["csv", "jsonl"], required=True
    )
    sp_imp.add_argument("--path", type=Path, required=True)
    sp_imp.add_argument("--bio-type", default="dna", choices=["dna", "protein"])
    sp_imp.add_argument("--alphabet", default="dna_4")
    sp_imp.set_defaults(func=cmd_import)

    sp_att = sp.add_parser(
        "attach",
        add_help=False,
        help="Attach namespaced columns to a dataset",
        description="Attach namespaced columns to a dataset",
    )
    _add_rich_help(sp_att)
    sp_att.add_argument("dataset")
    sp_att.add_argument("--path", type=Path, required=True)
    sp_att.add_argument("--namespace", required=True)
    sp_att.add_argument(
        "--id-col",
        default="id",
        help="Identifier column in the input. Use 'id' (sha1) or 'sequence'.",
    )
    sp_att.add_argument("--columns", default="")
    sp_att.add_argument("--allow-overwrite", action="store_true")
    sp_att.add_argument("--note", default="")
    sp_att.set_defaults(func=cmd_attach)

    sp_info = sp.add_parser(
        "info",
        add_help=False,
        help="Show dataset info",
        description="Show dataset info",
    )
    _add_rich_help(sp_info)
    sp_info.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Dataset name (optional if run inside a dataset folder)",
    )
    sp_info.set_defaults(func=cmd_info)

    sp_schema = sp.add_parser(
        "schema", add_help=False, help="Print schema", description="Print schema"
    )
    _add_rich_help(sp_schema)
    sp_schema.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Dataset name (optional if run inside a dataset folder)",
    )
    sp_schema.add_argument("--tree", action="store_true", help="Pretty tree view")
    sp_schema.set_defaults(func=cmd_schema)

    sp_head = sp.add_parser(
        "head",
        add_help=False,
        help="Show first N rows (pretty by default)",
        description="Show first N rows (pretty by default)",
    )
    _add_rich_help(sp_head)
    sp_head.add_argument(
        "target",
        nargs="?",
        default=".",
        help="Dataset name OR a filesystem path (defaults to current directory).",
    )
    sp_head.add_argument("-n", type=int, default=10)
    sp_head.add_argument("--raw", action="store_true", help="Disable pretty formatting")
    sp_head.add_argument("--max-colwidth", type=int, default=80)
    sp_head.add_argument("--max-list-items", type=int, default=6)
    sp_head.add_argument("--precision", type=int, default=4)
    sp_head.set_defaults(func=cmd_head)

    # lightweight parquet inspection (path-oriented)
    sp_cols = sp.add_parser(
        "cols",
        add_help=False,
        help="List columns for a Parquet file",
        description="List columns for a Parquet file",
    )
    _add_rich_help(sp_cols)
    sp_cols.add_argument(
        "path",
        nargs="?",
        default=Path("."),
        type=Path,
        help="Filesystem path (file or directory). Defaults to '.'.",
    )
    sp_cols.add_argument(
        "--glob",
        default=None,
        help="When path is a directory, limit matches (e.g. 'events*.parquet').",
    )
    sp_cols.set_defaults(func=cmd_cols)

    sp_desc = sp.add_parser(
        "describe",
        add_help=False,
        help="Column-wise summary (types, nulls, example values, list length stats)",
        description="Column-wise summary (types, nulls, example values, list length stats)",
    )
    _add_rich_help(sp_desc)
    sp_desc.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Dataset name (optional if run inside a dataset folder)",
    )
    sp_desc.add_argument("--columns", default="", help="CSV list of columns to include")
    sp_desc.add_argument(
        "--sample", type=int, default=1024, help="Sample size for examples/stats"
    )
    sp_desc.add_argument("--max-colwidth", type=int, default=80)
    sp_desc.add_argument("--max-list-items", type=int, default=6)
    sp_desc.add_argument("--precision", type=int, default=4)
    sp_desc.set_defaults(func=cmd_describe)

    sp_cell = sp.add_parser(
        "cell",
        add_help=False,
        help="Print a single cell from a Parquet file (path or dataset).",
        description="Print a single cell from a Parquet file (path or dataset).",
    )
    _add_rich_help(sp_cell)
    sp_cell.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Dataset name OR filesystem path (defaults to '.').",
    )
    sp_cell.add_argument(
        "--path", type=Path, required=False, help="Alternative to positional target."
    )
    sp_cell.add_argument("--row", type=int, required=True)
    sp_cell.add_argument("--col", required=True)
    sp_cell.add_argument("--glob", default=None)
    sp_cell.set_defaults(func=cmd_cell)

    sp_val = sp.add_parser(
        "validate",
        add_help=False,
        help="Validate dataset",
        description="Validate dataset",
    )
    _add_rich_help(sp_val)
    sp_val.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Dataset name (optional if run inside a dataset folder)",
    )
    sp_val.add_argument("--strict", action="store_true")
    sp_val.set_defaults(func=cmd_validate)

    sp_get = sp.add_parser(
        "get",
        add_help=False,
        help="Fetch a record by id",
        description="Fetch a record by id",
    )
    _add_rich_help(sp_get)
    sp_get.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Dataset name (optional if run inside a dataset folder)",
    )
    # Positional shorthand: allow `usr get <id>` when dataset is omitted
    sp_get.add_argument(
        "id_positional",
        nargs="?",
        default=None,
        help="Shorthand: record id when dataset is omitted",
    )
    sp_get.add_argument("--id", required=False, help="Record id (sha1)")
    sp_get.add_argument("--columns", default="")
    sp_get.set_defaults(func=cmd_get)

    sp_grep = sp.add_parser(
        "grep",
        add_help=False,
        help="Regex search over sequences",
        description="Regex search over sequences",
    )
    _add_rich_help(sp_grep)
    sp_grep.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Dataset name (optional if run inside a dataset folder)",
    )
    sp_grep.add_argument("--pattern", required=True)
    sp_grep.add_argument("--limit", type=int, default=20)
    sp_grep.set_defaults(func=cmd_grep)

    sp_exp = sp.add_parser(
        "export",
        add_help=False,
        help="Export dataset to CSV/JSONL",
        description="Export dataset to CSV/JSONL",
    )
    _add_rich_help(sp_exp)
    sp_exp.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Dataset name (optional if run inside a dataset folder)",
    )
    sp_exp.add_argument("--fmt", choices=["csv", "jsonl"], required=True)
    sp_exp.add_argument("--out", type=Path, required=True)
    sp_exp.set_defaults(func=cmd_export)

    sp_snap = sp.add_parser(
        "snapshot",
        add_help=False,
        help="Create a snapshot",
        description="Create a snapshot",
    )
    _add_rich_help(sp_snap)
    sp_snap.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="Dataset name (optional if run inside a dataset folder)",
    )
    sp_snap.set_defaults(func=cmd_snapshot)

    sp_dedupe = sp.add_parser(
        "dedupe-sequences",
        add_help=False,
        help="Remove case-insensitive duplicate sequences (same letters, different capitalization).",
        description="Remove case-insensitive duplicate sequences (same letters, different capitalization).",
    )
    _add_rich_help(sp_dedupe)
    sp_dedupe.add_argument("dataset")
    sp_dedupe.add_argument(
        "--policy",
        choices=["keep-first", "keep-last", "ask"],
        default="keep-first",
        help="Which record to keep inside each duplicate group.",
    )
    sp_dedupe.add_argument("--dry-run", action="store_true")
    sp_dedupe.add_argument("-y", "--yes", action="store_true")
    sp_dedupe.set_defaults(func=cmd_dedupe_sequences)

    # ---------------- make-mock ----------------
    sp_mock = sp.add_parser(
        "make-mock",
        add_help=False,
        help="Create a mock dataset (optionally from CSV) with demo columns",
        description="Create a mock dataset (optionally from CSV) with demo columns",
    )
    _add_rich_help(sp_mock)
    sp_mock.add_argument("dataset")
    sp_mock.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of rows (ignored if --from-csv is shorter)",
    )
    sp_mock.add_argument(
        "--length",
        type=int,
        default=60,
        help="DNA length for RANDOM sequences (ignored with --from-csv)",
    )
    sp_mock.add_argument(
        "--x-dim", type=int, default=512, help="Length of demo X vector"
    )
    sp_mock.add_argument("--y-dim", type=int, default=8, help="Length of demo Y vector")
    sp_mock.add_argument("--seed", type=int, default=7, help="Random seed")
    sp_mock.add_argument(
        "--namespace",
        default="demo",
        help="Namespace for derived columns (default 'demo')",
    )
    sp_mock.add_argument(
        "--from-csv",
        type=Path,
        default=None,
        help="Use sequences from a CSV (must have 'sequence' column)",
    )
    sp_mock.add_argument(
        "--force", action="store_true", help="Overwrite existing dataset if present"
    )
    sp_mock.set_defaults(func=cmd_make_mock)

    # ---------------- add-demo-cols ----------------
    sp_add = sp.add_parser(
        "add-demo-cols",
        add_help=False,
        help="Add demo vectors/labels to an existing dataset",
        description="Add demo vectors/labels to an existing dataset",
    )
    _add_rich_help(sp_add)
    sp_add.add_argument("dataset")
    sp_add.add_argument("--x-dim", type=int, default=512)
    sp_add.add_argument("--y-dim", type=int, default=8)
    sp_add.add_argument("--seed", type=int, default=7)
    sp_add.add_argument("--namespace", default="demo")
    sp_add.add_argument("--allow-overwrite", action="store_true")
    sp_add.set_defaults(func=cmd_add_demo)

    # ---------------- remotes management ----------------
    sp_r = sp.add_parser(
        "remotes",
        add_help=False,
        help="List/Add/Show SSH remotes",
        description="List/Add/Show SSH remotes",
    )
    _add_rich_help(sp_r)
    sp_r_sub = sp_r.add_subparsers(dest="r_cmd", required=False)

    sp_r_list = sp_r_sub.add_parser(
        "list",
        add_help=False,
        help="List configured remotes",
        description="List configured remotes",
    )
    _add_rich_help(sp_r_list)
    sp_r_list.set_defaults(func=cmd_remotes_list)

    sp_r_show = sp_r_sub.add_parser(
        "show",
        add_help=False,
        help="Show one remote config",
        description="Show one remote config",
    )
    _add_rich_help(sp_r_show)
    sp_r_show.add_argument("name")
    sp_r_show.set_defaults(func=cmd_remotes_show)

    sp_r_add = sp_r_sub.add_parser(
        "add",
        add_help=False,
        help="Add an SSH remote",
        description="Add an SSH remote",
    )
    _add_rich_help(sp_r_add)
    sp_r_add.add_argument("name")
    sp_r_add.add_argument("--type", default="ssh", choices=["ssh"])
    sp_r_add.add_argument("--host", required=True)
    sp_r_add.add_argument("--user", required=True)
    sp_r_add.add_argument("--base-dir", required=True)
    sp_r_add.add_argument("--ssh-key-env", default=None)
    sp_r_add.set_defaults(func=cmd_remotes_add)

    # ---------------- convert-legacy (PT -> fresh dataset) ----------------
    sp_conv = sp.add_parser(
        "convert-legacy",
        add_help=False,
        help="Convert legacy .pt (list[dict]) files into a NEW USR dataset (records.parquet).",
        description="Convert legacy .pt (list[dict]) files into a NEW USR dataset (records.parquet).",
    )
    _add_rich_help(sp_conv)
    sp_conv.add_argument("dataset", help="Name for the new USR dataset to create")
    sp_conv.add_argument(
        "--paths",
        nargs="+",
        type=Path,
        required=True,
        help="One or more .pt files or directories to scan recursively.",
    )
    sp_conv.add_argument(
        "--expected-length",
        type=int,
        default=None,
        help="Fail if sequences differ from this length (default: profile default).",
    )
    sp_conv.add_argument(
        "--plan",
        default=None,
        help="Value for densegen__plan (default: profile default 'sigma70_mid').",
    )
    sp_conv.add_argument(
        "--force",
        action="store_true",
        help="If dataset exists, destroy and recreate it.",
    )
    sp_conv.set_defaults(func=cmd_convert_legacy)

    # ---------------- MERGE DATASETS ----------------
    sp_merge = sp.add_parser(
        "merge-datasets",
        add_help=False,
        help="Merge rows from a source USR dataset into a destination dataset.",
        description="Merge rows from a source USR dataset into a destination dataset.",
    )
    _add_rich_help(sp_merge)
    sp_merge.add_argument("--dest", required=True, help="Destination dataset name")
    sp_merge.add_argument("--src", required=True, help="Source dataset name")

    mode = sp_merge.add_mutually_exclusive_group()
    mode.add_argument(
        "--require-same-columns",
        dest="require_same",
        action="store_true",
        help="Require identical column sets and types (strict).",
    )
    mode.add_argument(
        "--union-columns",
        dest="union_cols",
        action="store_true",
        help="Use column union (fill missing with NULLs). Default.",
    )

    sp_merge.add_argument(
        "--if-duplicate",
        dest="dup_policy",
        choices=["error", "skip", "prefer-src", "prefer-dest"],
        default="skip",
        help="Row duplicate policy by id (default: skip).",
    )
    sp_merge.add_argument(
        "--columns",
        default="",
        help="Optional CSV list of columns to keep (essentials always included).",
    )
    sp_merge.add_argument("--dry-run", action="store_true")
    sp_merge.add_argument("-y", "--yes", action="store_true", help="Assume yes")
    sp_merge.add_argument("--note", default="", help="Note to write into .events.log")
    sp_merge.set_defaults(func=cmd_merge_datasets)

    sp_merge.add_argument(
        "--coerce-overlap",
        choices=["none", "to-dest"],
        default="to-dest",
        help="If types differ on overlapping columns, attempt safe coercion to the "
        "destination column type (JSON->struct/list, numeric strings->numbers).",
    )

    sp_merge.add_argument(
        "--no-avoid-casefold-dups",
        dest="avoid_casefold_dups",
        action="store_false",
        help="Allow rows with the same letters (ignoring case) to be merged (NOT recommended).",
    )

    # ---------------- diff/status/pull/push ----------------
    def add_sync_common(p_: argparse.ArgumentParser):
        # dataset can be omitted if run inside a dataset dir under --root
        p_.add_argument("dataset", nargs="?", default=None)
        p_.add_argument("--remote", "--from", "--to", dest="remote", required=True)
        p_.add_argument(
            "-y", "--yes", action="store_true", help="Assume yes (no prompt)"
        )
        p_.add_argument("--dry-run", action="store_true")
        p_.add_argument("--primary-only", action="store_true")
        p_.add_argument("--skip-snapshots", action="store_true")
        p_.add_argument(
            "--remote-path",
            default=None,
            help="Override remote path (FILE mode). Absolute or relative to remote repo_root.",
        )
        p_.add_argument(
            "--repo-root",
            default=None,
            help="Local repo root to compute remote path (FILE mode). If omitted, use remote.local_repo_root or DNADESIGN_REPO_ROOT.",  # noqa
        )

    sp_diff = sp.add_parser(
        "diff",
        add_help=False,
        help="Show compact diff with a remote",
        description="Show compact diff with a remote",
    )
    _add_rich_help(sp_diff)
    add_sync_common(sp_diff)
    sp_diff.set_defaults(func=cmd_diff)

    sp_status = sp.add_parser(
        "status",
        add_help=False,
        help="Alias for diff (summary only)",
        description="Alias for diff (summary only)",
    )
    _add_rich_help(sp_status)
    add_sync_common(sp_status)
    sp_status.set_defaults(func=cmd_diff)

    sp_pull = sp.add_parser(
        "pull",
        add_help=False,
        help="Pull dataset from remote (safe overwrite)",
        description="Pull dataset from remote (safe overwrite)",
    )
    _add_rich_help(sp_pull)
    add_sync_common(sp_pull)
    sp_pull.set_defaults(func=cmd_pull)

    sp_push = sp.add_parser(
        "push",
        add_help=False,
        help="Push dataset to remote (safe overwrite)",
        description="Push dataset to remote (safe overwrite)",
    )
    _add_rich_help(sp_push)
    add_sync_common(sp_push)
    sp_push.set_defaults(func=cmd_push)

    # --------------- dispatch ----------------
    args = p.parse_args()
    try:
        args.func(args)
    except UserAbort:
        raise SystemExit(130)
    except SequencesError as e:
        _print_user_error(e)
        raise SystemExit(2)
    except FileExistsError as e:
        print(f"ERROR: {e}")
        raise SystemExit(3)
    except FileNotFoundError as e:
        # Path-oriented UX: no Python traceback for common "nothing here" cases
        print(f"ERROR: {e}")
        raise SystemExit(4)


# ---------- Rich help injection ----------
def _add_rich_help(parser: argparse.ArgumentParser) -> None:
    """Attach a Rich-styled -h/--help action to this parser."""
    parser.add_argument(
        "-h", "--help", action=_RichHelpAction, help="Show this message and exit."
    )


class _RichHelpAction(argparse.Action):
    """Argparse action to render help via Rich; falls back to default text if Rich not available."""

    def __init__(
        self,
        option_strings,
        dest=argparse.SUPPRESS,
        default=argparse.SUPPRESS,
        help=None,
    ):
        super().__init__(option_strings=option_strings, dest=dest, nargs=0, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        try:
            _print_help_rich(parser)
        except Exception:
            # Safe fallback to regular argparse help
            print(parser.format_help())
        parser.exit()


def _print_help_rich(parser: argparse.ArgumentParser) -> None:
    # Try to reuse the project's Rich loader to keep a consistent error if 'rich' is missing.
    from .ui import _require_rich

    console = _require_rich()
    from rich import box
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    # Header
    title = f"USR v{__version__} CLI — Parquet-backed datasets; SSH remotes sync."
    usage = parser.format_usage().strip().replace("usage:", "Usage:")

    console.print(
        Panel.fit(Text(title, style="bold"), box=box.ROUNDED, border_style="cyan")
    )
    console.print(Text(usage, style="bold"))
    console.print()

    # Options table
    opt_table = Table(
        title="Options",
        box=box.ROUNDED,
        show_lines=False,
        expand=False,
        header_style="bold cyan",
        border_style="dim",
    )
    opt_table.add_column("Flag", no_wrap=True)
    opt_table.add_column("Description")
    for act in parser._actions:
        # Skip subparsers themselves; they render below
        if isinstance(act, argparse._SubParsersAction):
            continue
        # Include only options (those that have flags)
        if not getattr(act, "option_strings", None):
            continue
        if not act.option_strings:
            continue
        flags = ", ".join(act.option_strings)
        takes_value = (getattr(act, "nargs", None) not in (0, None)) or (
            getattr(act, "metavar", None) and getattr(act, "nargs", None) not in (0,)
        )
        if takes_value:
            metavar = act.metavar or str(act.dest or "").upper()
            flags = f"{flags} {metavar}"
        desc = act.help or ""
        opt_table.add_row(flags, desc)
    console.print(opt_table)

    # Commands table (if any)
    sub_actions = [
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    ]
    if sub_actions:
        cmd_table = Table(
            title="Commands",
            box=box.ROUNDED,
            show_lines=False,
            expand=False,
            header_style="bold cyan",
            border_style="dim",
        )
        cmd_table.add_column("Command", no_wrap=True)
        cmd_table.add_column("Summary")
        for sub in sub_actions:
            for name, subparser in sorted(sub.choices.items(), key=lambda kv: kv[0]):
                summary = getattr(subparser, "description", "") or ""
                cmd_table.add_row(name, summary)
        console.print(cmd_table)
    console.print()


# ---------- helpers & command impls ----------
def list_datasets(root: Path):
    root = root.resolve()
    if not root.exists():
        return []
    return sorted(
        [
            p.name
            for p in root.iterdir()
            if p.is_dir() and (p / "records.parquet").exists()
        ]
    )


def cmd_ls(args):
    names = list_datasets(args.root)
    if not names:
        print(f"(no datasets under {args.root})")
        return

    def _fmt_bytes(n: int | None) -> str:
        if not isinstance(n, int):
            return "?"
        units = ["B", "KB", "MB", "GB", "TB", "PB"]
        i = 0
        x = float(n)
        while x >= 1024 and i < len(units) - 1:
            x /= 1024.0
            i += 1
        return f"{x:.0f}{units[i]}"

    rows = []
    for name in names:
        rp = args.root / name / "records.parquet"
        try:
            pf = pq.ParquetFile(str(rp))
            st = rp.stat()
            rows.append(
                {
                    "dataset": name,
                    "rows": pf.metadata.num_rows,
                    "cols": pf.metadata.num_columns,
                    "size": _fmt_bytes(int(st.st_size)),
                    "updated": _dt.datetime.fromtimestamp(int(st.st_mtime)).isoformat(
                        timespec="seconds"
                    ),
                }
            )
        except Exception:
            rows.append(
                {
                    "dataset": name,
                    "rows": None,
                    "cols": None,
                    "size": "?",
                    "updated": "?",
                }
            )

    df = pd.DataFrame(rows, columns=["dataset", "rows", "cols", "size", "updated"])
    if getattr(args, "rich", False):
        render_table_rich(df, title="USR datasets", caption=str(args.root))
    else:
        print_df_plain(df)


def cmd_init(args):
    d = Dataset(args.root, args.dataset)
    d.init(source=args.source, notes=args.notes)
    print(f"Initialized dataset at {d.records_path}")


def cmd_import(args):
    d = Dataset(args.root, args.dataset)
    in_path = _resolve_path_anywhere(args.path)
    if args.source_format == "csv":
        df = pd.read_csv(in_path)
    else:
        df = pd.read_json(in_path, lines=True)
    n = d.import_rows(
        df,
        default_bio_type=args.bio_type,
        default_alphabet=args.alphabet,
        source=str(in_path),
    )
    print(f"Imported {n} records into {d.name}")
    try:
        cmd = f"usr import {args.dataset} --from {args.source_format} --path {in_path} --bio-type {args.bio_type} --alphabet {args.alphabet}"  # noqa
        d.append_meta_note(f"Imported {n} records from {in_path}", cmd)
    except Exception:
        pass


def cmd_attach(args):
    d = Dataset(args.root, args.dataset)
    in_path = _resolve_path_anywhere(args.path)
    cols = [c.strip() for c in args.columns.split(",")] if args.columns else None
    n = d.attach_columns(  # friendly alias
        in_path,
        namespace=args.namespace,
        id_col=args.id_col,
        columns=cols,
        allow_overwrite=bool(args.allow_overwrite),
        note=args.note,
    )
    print(
        f"Attached {n} matched row(s) of {args.namespace} columns into {d.name} "
        f"(input file: {in_path})"
    )
    try:
        cols = args.columns or "(all columns)"
        cmd = f'usr attach {args.dataset} --path {in_path} --namespace {args.namespace} --id-col {args.id_col} --columns "{cols}"'  # noqa
        d.append_meta_note(
            f"Attached columns under '{args.namespace}' ({n} row match)", cmd
        )
    except Exception:
        pass


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
    ds_name = _resolve_dataset_name_interactive(
        args.root, getattr(args, "dataset", None), bool(getattr(args, "rich", False))
    )
    if not ds_name:
        return
    d = Dataset(args.root, ds_name)
    info = d.info()
    for k, v in info.items():
        print(f"{k}: {v}")


def cmd_schema(args):
    ds_name = _resolve_dataset_name_interactive(
        args.root, getattr(args, "dataset", None), bool(getattr(args, "rich", False))
    )
    if not ds_name:
        return
    d = Dataset(args.root, ds_name)
    sch = d.schema()
    if args.tree:
        if getattr(args, "rich", False):
            lines = render_schema_tree(sch).splitlines()
            render_schema_tree_rich(lines, title=str(d.records_path))
        else:
            print(render_schema_tree(sch))
    else:
        print(sch)


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
        try:
            return fmt_value(x, opts)
        except Exception:
            return str(x)

    return df.applymap(_fmt_cell)


def _log_implicit_pick_if_dataset(pq_path: Path | None, reason: str) -> None:
    """
    If the selected Parquet file lives inside a USR dataset folder (has meta.md),
    append a small event so the implicit behavior is auditable.
    """
    if pq_path is None:
        return
    try:
        p = Path(pq_path).resolve()
        d = p.parent
        if (d / "meta.md").exists():
            append_event(
                d / ".events.log",
                {
                    "action": "implicit_file_pick",
                    "path": str(p),
                    "cwd": str(Path.cwd().resolve()),
                    "reason": reason,
                },
            )
    except Exception:
        # Never block UX on logging failures
        pass


def cmd_head(args):
    # Path-first mode (file/dir), then dataset fallback.
    p = Path(args.target)
    implicit = str(getattr(args, "target", "")) in {"", ".", "./"}
    if p.exists():
        pq_path = _select_parquet_target_interactive(
            p,
            glob=None,
            use_rich=bool(getattr(args, "rich", False)),
            confirm_if_inferred=implicit,
        )
        if pq_path is None:
            return
        if implicit:
            _log_implicit_pick_if_dataset(pq_path, reason="head")
        tbl = pq.read_table(pq_path)
        df = tbl.to_pandas().head(int(args.n))
        caption = f"{pq_path}  rows={tbl.num_rows:,}  cols={tbl.num_columns}"
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
    # dataset mode
    d = Dataset(args.root, args.target)
    df = d.head(args.n)
    try:
        meta = pq.ParquetFile(str(d.records_path)).metadata
        caption = f"{d.records_path}  rows={meta.num_rows:,}  cols={meta.num_columns}"
    except Exception:
        caption = str(d.records_path)
    if getattr(args, "rich", False):
        if not args.raw:
            df = _pretty_df(df, _pretty_opts_from(args))
        render_table_rich(
            df,
            title=f"dataset: {args.target}",
            caption=caption,
            max_colwidth=int(args.max_colwidth),
        )
    else:
        if not args.raw:
            df = _pretty_df(df, _pretty_opts_from(args))
        _print_df(df)


def cmd_cols(args):
    target = Path(getattr(args, "path", Path(".")))
    implicit = str(getattr(args, "path", Path("."))) in {".", "./"}
    pq_path = _select_parquet_target_interactive(
        target,
        glob=args.glob,
        use_rich=bool(getattr(args, "rich", False)),
        confirm_if_inferred=implicit,
    )
    if pq_path is None:
        return
    if implicit:
        _log_implicit_pick_if_dataset(pq_path, reason="cols")
    pf = pq.ParquetFile(str(pq_path))
    fields = []
    sch = pf.schema_arrow
    for i in range(len(sch.names)):
        f = sch.field(i)
        fields.append(
            {"name": f.name, "type": str(f.type), "nullable": bool(f.nullable)}
        )
    df = pd.DataFrame(fields, columns=["name", "type", "nullable"])
    caption = (
        f"{pq_path}  rows={pf.metadata.num_rows:,}  cols={pf.metadata.num_columns}"
    )
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
    tbl = pq.read_table(d.records_path)
    cols = (
        [c.strip() for c in args.columns.split(",") if c.strip()]
        if args.columns
        else None
    )
    opts = PrettyOpts(
        max_colwidth=int(args.max_colwidth),
        max_list_items=int(args.max_list_items),
        precision=int(args.precision),
    )
    prof = profile_table(tbl, opts, columns=cols, sample=int(args.sample))
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
        caption = f"{d.records_path}  rows={tbl.num_rows:,}  cols={tbl.num_columns}"
        render_table_rich(
            df,
            title=f"describe: {ds_name}",
            caption=caption,
            max_colwidth=int(args.max_colwidth),
        )
    else:
        print(f"# {d.records_path}  rows={tbl.num_rows:,}  cols={tbl.num_columns}")
        _print_df(df)


def cmd_cell(args):
    # Determine target path: --path, or positional, or dataset name
    path_arg = getattr(args, "path", None)
    target_arg = getattr(args, "target", None)
    implicit = (path_arg is None) and (target_arg is None)
    tgt = path_arg or target_arg or "."
    p = Path(tgt)
    pq_path: Path | None = None
    if p.exists():
        pq_path = _select_parquet_target_interactive(
            p, glob=args.glob, use_rich=False, confirm_if_inferred=implicit
        )
        if pq_path is None:
            return
        if implicit:
            _log_implicit_pick_if_dataset(pq_path, reason="cell")
    else:
        try:
            d = Dataset(args.root, str(tgt))
            pq_path = d.records_path
        except Exception:
            pq_path = None
    if pq_path is None or not Path(pq_path).exists():
        print(f"Target not found: {tgt}")
        print(
            "Tip: pass a file/dir path, or a dataset name (e.g., 'usr cell demo --row 0 --col sequence')."
        )
        return
    col = str(args.col)
    row = int(args.row)
    try:
        tbl = pq.read_table(pq_path, columns=[col])
    except KeyError:
        pf = pq.ParquetFile(str(pq_path))
        names = ", ".join(pf.schema_arrow.names)
        print(f"Column '{col}' not found in {pq_path}.")
        print(f"Available columns: {names}")
        return
    if row < 0 or row >= tbl.num_rows:
        print(f"Row {row} out of range (0..{tbl.num_rows-1}).")
        return
    val = tbl.column(0)[row].as_py()
    print(val)


def cmd_validate(args):
    ds_name = _resolve_dataset_name_interactive(
        args.root, getattr(args, "dataset", None), False
    )
    if not ds_name:
        return
    d = Dataset(args.root, ds_name)
    d.validate(strict=bool(getattr(args, "strict", False)))
    print("OK: validation passed.")


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
    df = d.get(rid, columns=cols)
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
    df = d.grep(args.pattern, args.limit)
    if getattr(args, "rich", False):
        df_fmt = df.applymap(lambda x: fmt_value(x, PrettyOpts()))
        render_table_rich(df_fmt, title=f"grep: {args.pattern}")
    else:
        _print_df(df)


def cmd_export(args):
    ds_name = _resolve_dataset_name_interactive(
        args.root, getattr(args, "dataset", None), False
    )
    if not ds_name:
        return
    d = Dataset(args.root, ds_name)
    d.export(args.fmt, args.out)
    print(f"Wrote {args.out}")


def cmd_snapshot(args):
    ds_name = _resolve_dataset_name_interactive(
        args.root, getattr(args, "dataset", None), False
    )
    if not ds_name:
        return
    d = Dataset(args.root, ds_name)
    d.snapshot()
    print(f"Snapshot saved under {d.snapshot_dir}")
    try:
        d.append_meta_note("Snapshot saved", f"usr snapshot {ds_name}")
    except Exception:
        pass


def cmd_convert_legacy(args):
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
    try:
        src = (
            f"--from-csv {spec.csv_path}"
            if spec.csv_path
            else f"--length {spec.length}"
        )
        cmd = f"usr make-mock {args.dataset} --n {spec.n} {src} --namespace {spec.namespace} --x-dim {spec.x_dim} --y-dim {spec.y_dim}"  # noqa
        Dataset(args.root, args.dataset).append_meta_note("Created mock dataset", cmd)
    except Exception:
        pass


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
    try:
        cmd = f"usr add-demo-cols {args.dataset} --x-dim {args.x_dim} --y-dim {args.y_dim} --namespace {args.namespace}"
        Dataset(args.root, args.dataset).append_meta_note("Added demo columns", cmd)
    except Exception:
        pass


# ---------- MERGE DATASETS ----------
def _policy_from_str(s: str) -> MergePolicy:
    return {
        "error": MergePolicy.ERROR,
        "skip": MergePolicy.SKIP,
        "prefer-src": MergePolicy.PREFER_SRC,
        "prefer-dest": MergePolicy.PREFER_DEST,
    }[s]


def cmd_merge_datasets(args):
    cols_subset = (
        [c.strip() for c in args.columns.split(",") if c.strip()]
        if args.columns
        else None
    )
    mode = (
        MergeColumnsMode.REQUIRE_SAME if args.require_same else MergeColumnsMode.UNION
    )
    policy = _policy_from_str(args.dup_policy)

    preview = merge_usr_to_usr(
        root=args.root,
        dest=args.dest,
        src=args.src,
        columns_mode=mode,
        duplicate_policy=policy,
        columns_subset=cols_subset,
        dry_run=bool(args.dry_run),
        assume_yes=bool(args.yes),
        note=args.note or "",
        overlap_coercion=("to-dest" if args.coerce_overlap == "to-dest" else "none"),
        avoid_casefold_dups=bool(getattr(args, "avoid_casefold_dups", True)),
    )

    action = "DRY-RUN" if args.dry_run else "MERGED"
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
    if not args.dry_run:
        try:
            cmd = (
                f"usr merge-datasets --dest {args.dest} --src {args.src} "
                f"{'--require-same-columns' if args.require_same else '--union-columns'} "
                f"--if-duplicate {args.dup_policy}"
            )
            Dataset(args.root, args.dest).append_meta_note(
                f"Merged from '{args.src}' → '{args.dest}' (added {preview.new_rows} rows; dup_policy={preview.duplicate_policy.value})",  # noqa
                cmd,
            )
        except Exception:
            pass


# ---------- remotes commands ----------
def cmd_remotes_list(args):
    remotes = load_all()
    if not remotes:
        print("(no remotes configured)")
        return
    for name, cfg in remotes.items():
        print(f"{name:20s} ssh {cfg.user}@{cfg.host}  base_dir={cfg.base_dir}")


def cmd_remotes_show(args):
    cfg = get_remote(args.name)
    print(f"name     : {cfg.name}")
    print("type     : ssh")
    print(f"ssh      : {cfg.user}@{cfg.host}")
    print(f"base_dir : {cfg.base_dir}")
    print(f"ssh_key  : {cfg.ssh_key_env or '(ssh-agent or default key)'}")


def cmd_remotes_add(args):
    if args.type != "ssh":
        raise SystemExit("Only --type ssh is supported.")
    cfg = SSHRemoteConfig(
        name=args.name,
        host=args.host,
        user=args.user,
        base_dir=args.base_dir,
        ssh_key_env=args.ssh_key_env,
    )
    path = save_remote(cfg)
    print(f"Saved remote '{cfg.name}' to {path}")


# ---------- diff/pull/push ----------
def _print_diff(summary, use_rich: bool | None = None):
    def fmt_sz(n):
        if n is None:
            return "?"
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if n < 1024:
                return f"{n:.0f}{unit}"
            n /= 1024
        return f"{n:.0f}PB"

    if use_rich:
        render_diff_rich(summary)
        return
    pl, pr = summary.primary_local, summary.primary_remote
    print(f"Dataset: {summary.dataset}")

    local_line = (
        f"Local  : {pl.sha256 or '?'}  "
        f"size={fmt_sz(pl.size)}  "
        f"rows={pl.rows or '?'}  "
        f"cols={pl.cols or '?'}"
    )
    print(local_line)

    remote_line = (
        f"Remote : {pr.sha256 or '?'}  "
        f"size={fmt_sz(pr.size)}  "
        f"rows={pr.rows or '?'}  "
        f"cols={pr.cols or '?'}"
    )
    print(remote_line)

    eq = "==" if (pl.sha256 and pr.sha256 and pl.sha256 == pr.sha256) else "≠"
    print(f"Primary sha: {pl.sha256 or '?'} {eq} {pr.sha256 or '?'}")
    print(
        "meta.md     mtime: "
        f"{summary.meta_local_mtime or '-'}  →  {summary.meta_remote_mtime or '-'}"
    )
    delta_evt = max(0, summary.events_remote_lines - summary.events_local_lines)
    print(
        ".events.log lines: "
        f"local={summary.events_local_lines}  "
        f"remote={summary.events_remote_lines}  "
        f"(+{delta_evt} on remote)"
    )
    print(
        "_snapshots : "
        f"remote_count={summary.snapshots.count}  "
        f"newer_than_local={summary.snapshots.newer_than_local}"
    )
    print("Status     :", "CHANGES" if summary.has_change else "up-to-date")


def _confirm_or_abort(summary, assume_yes: bool):
    if not summary.has_change:
        print("Already up to date.")
        return
    if assume_yes:
        return
    print("\nChanges detected. Proceed with overwrite?")
    ans = input("[Enter = Yes / n = No] : ").strip().lower()
    if ans in {"n", "no"}:
        raise UserAbort("User cancelled.")


def _opts_from_args(args) -> SyncOptions:
    return SyncOptions(
        primary_only=bool(args.primary_only),
        skip_snapshots=bool(args.skip_snapshots),
        dry_run=bool(args.dry_run),
        assume_yes=bool(args.yes),
    )


def _is_file_mode_target(target: str | None) -> bool:
    """Heuristic: treat as FILE mode only if the user passed something path-like explicitly."""
    if not target:
        return False
    try:
        p = Path(target)
    except Exception:
        return False
    return ("/" in target) or target.endswith(".parquet") or p.exists()


def cmd_diff(args):
    target = args.dataset  # may be None
    if _is_file_mode_target(target):
        local_file = Path(target).resolve()
        if local_file.is_dir():
            raise SystemExit("FILE mode: pass a file path, not a directory.")
        remote_path = _resolve_remote_path_for_file(local_file, args)
        s = plan_diff_file(local_file, args.remote, remote_path=remote_path)
    else:
        ds_name = _resolve_dataset_name_interactive(
            args.root,
            getattr(args, "dataset", None),
            bool(getattr(args, "rich", False)),
        )
        if not ds_name:
            return
        s = plan_diff(args.root, ds_name, args.remote)
    _print_diff(s, use_rich=bool(getattr(args, "rich", False)))


def cmd_pull(args):
    target = args.dataset
    if _is_file_mode_target(target):
        if args.primary_only or args.skip_snapshots:
            raise SystemExit(
                "--primary-only/--skip-snapshots are dataset-only flags (not valid in FILE mode)."
            )
        local_file = Path(target).resolve()
        if local_file.is_dir():
            raise SystemExit("FILE mode: pass a file path, not a directory.")
        remote_path = _resolve_remote_path_for_file(local_file, args)
        s = plan_diff_file(local_file, args.remote, remote_path=remote_path)
        _print_diff(s, use_rich=bool(getattr(args, "rich", False)))
        _confirm_or_abort(s, assume_yes=bool(args.yes))
        execute_pull_file(local_file, args.remote, remote_path, _opts_from_args(args))
    else:
        ds_name = _resolve_dataset_name_interactive(
            args.root,
            getattr(args, "dataset", None),
            bool(getattr(args, "rich", False)),
        )
        if not ds_name:
            return
        s = plan_diff(args.root, ds_name, args.remote)
        _print_diff(s, use_rich=bool(getattr(args, "rich", False)))
        _confirm_or_abort(s, assume_yes=bool(args.yes))
        execute_pull(args.root, ds_name, args.remote, _opts_from_args(args))
    if not args.dry_run:
        print("Pull complete.")


def cmd_push(args):
    target = args.dataset
    if _is_file_mode_target(target):
        if args.primary_only or args.skip_snapshots:
            raise SystemExit(
                "--primary-only/--skip-snapshots are dataset-only flags (not valid in FILE mode)."
            )
        local_file = Path(target).resolve()
        if local_file.is_dir():
            raise SystemExit("FILE mode: pass a file path, not a directory.")
        remote_path = _resolve_remote_path_for_file(local_file, args)
        s = plan_diff_file(local_file, args.remote, remote_path=remote_path)
        _print_diff(s, use_rich=bool(getattr(args, "rich", False)))
        _confirm_or_abort(s, assume_yes=bool(args.yes))
        execute_push_file(local_file, args.remote, remote_path, _opts_from_args(args))
    else:
        ds_name = _resolve_dataset_name_interactive(
            args.root,
            getattr(args, "dataset", None),
            bool(getattr(args, "rich", False)),
        )
        if not ds_name:
            return
        s = plan_diff(args.root, ds_name, args.remote)
        _print_diff(s, use_rich=bool(getattr(args, "rich", False)))
        _confirm_or_abort(s, assume_yes=bool(args.yes))
        execute_push(args.root, ds_name, args.remote, _opts_from_args(args))
    if not args.dry_run:
        print("Push complete.")


# ---------- helpers for FILE mode ----------
def _resolve_remote_path_for_file(local_file: Path, args) -> str:
    if args.remote_path:
        return args.remote_path
    cfg = get_remote(args.remote)
    if not cfg.repo_root:
        raise SystemExit(
            "FILE mode requires remote.repo_root in remotes.yaml or --remote-path."
        )
    import os

    local_root = (
        args.repo_root or cfg.local_repo_root or os.environ.get("DNADESIGN_REPO_ROOT")
    )
    if not local_root:
        raise SystemExit(
            "FILE mode requires a local repo root. Pass --repo-root, set DNADESIGN_REPO_ROOT, or add local_repo_root in remotes.yaml."  # noqa
        )
    try:
        rel = local_file.resolve().relative_to(Path(local_root).resolve())
    except Exception:
        raise SystemExit(
            f"Cannot compute path relative to local repo root: {local_file} not under {local_root}"
        )
    from pathlib import PurePosixPath

    return str(PurePosixPath(cfg.repo_root).joinpath(rel.as_posix()))


def cmd_dedupe_sequences(args):
    import pyarrow as pa
    import pyarrow.compute as pc

    from .io import append_event as _append_event
    from .io import read_parquet, write_parquet_atomic

    d = Dataset(args.root, args.dataset)
    tbl = read_parquet(d.records_path)
    if tbl.num_rows == 0:
        print("Dataset is empty.")
        return

    # For deterministic ordering, prefer created_at then id
    df = tbl.select(["id", "bio_type", "sequence", "created_at"]).to_pandas()
    df["_key"] = df["bio_type"].str.lower() + "|" + df["sequence"].str.upper()

    groups = df.groupby("_key").agg({"id": "count"})
    dup_keys = groups[groups["id"] > 1].index.tolist()
    if not dup_keys:
        print("OK: no case-insensitive duplicate sequences found.")
        return

    # Decide which id to keep per group
    keep_ids: list[str] = []
    drop_ids: list[str] = []

    use_rich = bool(getattr(args, "rich", False))

    def _preview_group(seq_letters: str, g_sorted) -> None:
        rows = []
        for i, r in g_sorted.reset_index(drop=True).iterrows():
            rows.append({"#": i + 1, "id": r["id"], "created_at": str(r["created_at"])})
        dfp = pd.DataFrame(rows, columns=["#", "id", "created_at"])
        if use_rich:
            render_table_rich(dfp, title=f"duplicate: {seq_letters}")
        else:
            print(f"• {seq_letters}")
            print_df_plain(dfp)

    for k, g in df[df["_key"].isin(dup_keys)].groupby("_key"):
        seq_letters = k.split("|", 1)[1]
        g_sorted = g.sort_values(
            ["created_at", "id"],
            ascending=True if args.policy in {"keep-first", "ask"} else False,
            kind="stable",
        )
        if args.policy == "ask" and not args.dry_run:
            _preview_group(seq_letters, g_sorted)
            ans = (
                input("Keep which row? [1..n], 0 = drop all, s = skip group: ")
                .strip()
                .lower()
            )
            if ans in {"s", "skip"}:
                keep_ids.extend(g_sorted["id"].tolist())
                continue
            if ans in {"0", "drop-all"}:
                drop_ids.extend(g_sorted["id"].tolist())
                continue
            try:
                kidx = int(ans)
                if 1 <= kidx <= len(g_sorted):
                    keep_ids.append(g_sorted.iloc[kidx - 1]["id"])
                    drop_ids.extend(
                        g_sorted.drop(g_sorted.index[kidx - 1])["id"].tolist()
                    )
                    continue
            except Exception:
                pass
            keep_ids.append(g_sorted.iloc[0]["id"])
            drop_ids.extend(g_sorted.iloc[1:]["id"].tolist())
        else:
            keep_ids.append(g_sorted.iloc[0]["id"])
            drop_ids.extend(g_sorted.iloc[1:]["id"].tolist())

    print(
        f"Found {len(dup_keys)} duplicate group(s); would drop {len(drop_ids)} row(s)."
    )
    for k, g in df[df["_key"].isin(dup_keys)].groupby("_key"):
        print("•", k.split("|", 1)[1])
        print(
            "   keep:",
            (
                g.sort_values("created_at").iloc[0]["id"]
                if args.policy != "keep-last"
                else g.sort_values("created_at", ascending=False).iloc[0]["id"]
            ),
        )
        if len(g) > 1:
            print(
                "   drop:",
                ", ".join(
                    g.sort_values("created_at")
                    .iloc[1:]["id"]
                    .map(lambda s: s[:8] + "…")
                    .tolist()
                ),
            )
        if len(dup_keys) > 5:
            print("   …")
            break

    if args.dry_run:
        return
    if not args.yes:
        ans = input("Proceed with destructive de-duplication? [y/N]: ").strip().lower()
        if ans not in {"y", "yes"}:
            print("Aborted.")
            return

    drop_set = set(drop_ids)
    if drop_set:
        drop_mask = pc.is_in(tbl.column("id"), value_set=pa.array(list(drop_set)))
        new_tbl = tbl.filter(pc.invert(drop_mask))
    else:
        new_tbl = tbl
    write_parquet_atomic(
        new_tbl, d.records_path, d.snapshot_dir, preserve_metadata_from=tbl
    )
    _append_event(
        d.events_path,
        {
            "action": "dedupe_sequences",
            "dataset": d.name,
            "groups": len(dup_keys),
            "rows_dropped": len(drop_ids),
            "policy": args.policy,
        },
    )
    print(
        f"[dedupe] dropped {len(drop_ids)} row(s); dataset now has {new_tbl.num_rows} rows."
    )
