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
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .config import SSHRemoteConfig, get_remote, load_all, save_remote
from .convert_legacy import convert_legacy, profile_60bp_dual_promoter
from .dataset import Dataset
from .errors import SequencesError, UserAbort
from .merge_datasets import (
    MergeColumnsMode,
    MergePolicy,
    merge_usr_to_usr,
)
from .mock import MockSpec, add_demo_columns, create_mock_dataset
from .sync import SyncOptions, execute_pull, execute_push, plan_diff

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
    p = argparse.ArgumentParser(
        prog="usr",
        description="USR CLI (Parquet-backed datasets; includes SSH remotes sync).",
    )
    default_root = (_pkg_usr_root() / "datasets").resolve()
    p.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help=f"Datasets root folder (defaults to {default_root})",
    )
    sp = p.add_subparsers(dest="cmd", required=True)

    # ---------------- core dataset commands ----------------
    sp_ls = sp.add_parser("ls", help="List datasets under root")
    sp_ls.set_defaults(func=cmd_ls)

    sp_init = sp.add_parser("init", help="Create an empty dataset")
    sp_init.add_argument("dataset")
    sp_init.add_argument("--source", default="")
    sp_init.add_argument("--notes", default="")
    sp_init.set_defaults(func=cmd_init)

    sp_imp = sp.add_parser("import", help="Import sequences into a dataset")
    sp_imp.add_argument("dataset")
    sp_imp.add_argument(
        "--from", dest="source_format", choices=["csv", "jsonl"], required=True
    )
    sp_imp.add_argument("--path", type=Path, required=True)
    sp_imp.add_argument("--bio-type", default="dna", choices=["dna", "protein"])
    sp_imp.add_argument("--alphabet", default="dna_4")
    sp_imp.set_defaults(func=cmd_import)

    sp_att = sp.add_parser("attach", help="Attach namespaced columns to a dataset")
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

    sp_info = sp.add_parser("info", help="Show dataset info")
    sp_info.add_argument("dataset")
    sp_info.set_defaults(func=cmd_info)

    sp_schema = sp.add_parser("schema", help="Print schema")
    sp_schema.add_argument("dataset")
    sp_schema.set_defaults(func=cmd_schema)

    sp_head = sp.add_parser("head", help="Show first N rows")
    sp_head.add_argument("dataset")
    sp_head.add_argument("-n", type=int, default=10)
    sp_head.set_defaults(func=cmd_head)

    sp_val = sp.add_parser("validate", help="Validate dataset")
    sp_val.add_argument("dataset")
    sp_val.add_argument("--strict", action="store_true")
    sp_val.set_defaults(func=cmd_validate)

    sp_get = sp.add_parser("get", help="Fetch a record by id")
    sp_get.add_argument("dataset")
    sp_get.add_argument("--id", required=True)
    sp_get.add_argument("--columns", default="")
    sp_get.set_defaults(func=cmd_get)

    sp_grep = sp.add_parser("grep", help="Regex search over sequences")
    sp_grep.add_argument("dataset")
    sp_grep.add_argument("--pattern", required=True)
    sp_grep.add_argument("--limit", type=int, default=20)
    sp_grep.set_defaults(func=cmd_grep)

    sp_exp = sp.add_parser("export", help="Export dataset to CSV/JSONL")
    sp_exp.add_argument("dataset")
    sp_exp.add_argument("--fmt", choices=["csv", "jsonl"], required=True)
    sp_exp.add_argument("--out", type=Path, required=True)
    sp_exp.set_defaults(func=cmd_export)

    sp_snap = sp.add_parser("snapshot", help="Create a snapshot")
    sp_snap.add_argument("dataset")
    sp_snap.set_defaults(func=cmd_snapshot)

    # ---------------- make-mock ----------------
    sp_mock = sp.add_parser(
        "make-mock",
        help="Create a mock dataset (optionally from CSV) with demo columns",
    )
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
        "add-demo-cols", help="Add demo vectors/labels to an existing dataset"
    )
    sp_add.add_argument("dataset")
    sp_add.add_argument("--x-dim", type=int, default=512)
    sp_add.add_argument("--y-dim", type=int, default=8)
    sp_add.add_argument("--seed", type=int, default=7)
    sp_add.add_argument("--namespace", default="demo")
    sp_add.add_argument("--allow-overwrite", action="store_true")
    sp_add.set_defaults(func=cmd_add_demo)

    # ---------------- remotes management ----------------
    sp_r = sp.add_parser("remotes", help="List/Add/Show SSH remotes")
    sp_r_sub = sp_r.add_subparsers(dest="r_cmd", required=False)

    sp_r_list = sp_r_sub.add_parser("list", help="List configured remotes")
    sp_r_list.set_defaults(func=cmd_remotes_list)

    sp_r_show = sp_r_sub.add_parser("show", help="Show one remote config")
    sp_r_show.add_argument("name")
    sp_r_show.set_defaults(func=cmd_remotes_show)

    sp_r_add = sp_r_sub.add_parser("add", help="Add an SSH remote")
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
        help="Convert legacy .pt (list[dict]) files into a NEW USR dataset (records.parquet).",
    )
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
        help="Merge rows from a source USR dataset into a destination dataset.",
    )
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

    # ---------------- diff/status/pull/push ----------------
    def add_sync_common(p_: argparse.ArgumentParser):
        p_.add_argument("dataset")
        p_.add_argument("--remote", "--from", "--to", dest="remote", required=True)
        p_.add_argument(
            "-y", "--yes", action="store_true", help="Assume yes (no prompt)"
        )
        p_.add_argument("--dry-run", action="store_true")
        p_.add_argument("--primary-only", action="store_true")
        p_.add_argument("--skip-snapshots", action="store_true")

    sp_diff = sp.add_parser("diff", help="Show compact diff with a remote")
    add_sync_common(sp_diff)
    sp_diff.set_defaults(func=cmd_diff)

    sp_status = sp.add_parser("status", help="Alias for diff (summary only)")
    add_sync_common(sp_status)
    sp_status.set_defaults(func=cmd_diff)

    sp_pull = sp.add_parser("pull", help="Pull dataset from remote (safe overwrite)")
    add_sync_common(sp_pull)
    sp_pull.set_defaults(func=cmd_pull)

    sp_push = sp.add_parser("push", help="Push dataset to remote (safe overwrite)")
    add_sync_common(sp_push)
    sp_push.set_defaults(func=cmd_push)

    # --------------- dispatch ----------------
    args = p.parse_args()
    try:
        args.func(args)
    except UserAbort:
        raise SystemExit(130)
    except SequencesError as e:
        print(f"ERROR: {e}")
        raise SystemExit(2)
    except FileExistsError as e:
        print(f"ERROR: {e}")
        raise SystemExit(3)


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
    ds = list_datasets(args.root)
    if not ds:
        print(f"(no datasets under {args.root})")
        return
    for name in ds:
        rp = args.root / name / "records.parquet"
        try:
            tbl = pq.read_table(rp, columns=["id"])
            print(f"{name:40s}  rows={tbl.num_rows:,}")
        except Exception:
            print(f"{name:40s}  rows=(unreadable)")


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


def cmd_info(args):
    d = Dataset(args.root, args.dataset)
    info = d.info()
    for k, v in info.items():
        print(f"{k}: {v}")


def cmd_schema(args):
    d = Dataset(args.root, args.dataset)
    print(d.schema())


def cmd_head(args):
    d = Dataset(args.root, args.dataset)
    df = d.head(args.n)
    pd.set_option("display.max_colwidth", 120)
    print(df)


def cmd_validate(args):
    d = Dataset(args.root, args.dataset)
    d.validate(strict=args.strict)
    print("OK: validation passed.")


def cmd_get(args):
    d = Dataset(args.root, args.dataset)
    cols = [c.strip() for c in args.columns.split(",")] if args.columns else None
    df = d.get(args.id, columns=cols)
    print("Not found." if df.empty else df.to_string(index=False))


def cmd_grep(args):
    d = Dataset(args.root, args.dataset)
    df = d.grep(args.pattern, args.limit)
    print(df.to_string(index=False))


def cmd_export(args):
    d = Dataset(args.root, args.dataset)
    d.export(args.fmt, args.out)
    print(f"Wrote {args.out}")


def cmd_snapshot(args):
    d = Dataset(args.root, args.dataset)
    d.snapshot()
    print(f"Snapshot saved under {d.snapshot_dir}")
    try:
        d.append_meta_note("Snapshot saved", f"usr snapshot {args.dataset}")
    except Exception:
        pass


def cmd_convert_legacy(args):
    # Resolve each provided path robustly (supports repo-relative and package-relative)
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

    msg = (
        f"Converted {stats.rows} row(s) from {stats.files} file(s) "
        f"into dataset '{args.dataset}'."
    )
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
    )

    # Always print a compact summary
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


def _print_diff(summary):
    def fmt_sz(n):
        if n is None:
            return "?"
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if n < 1024:
                return f"{n:.0f}{unit}"
            n /= 1024
        return f"{n:.0f}PB"

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


def cmd_diff(args):
    s = plan_diff(args.root, args.dataset, args.remote)
    _print_diff(s)


def _confirm_or_abort(summary, assume_yes: bool):
    if not summary.has_change:
        print("Already up to date.")
        return
    if assume_yes:
        return
    print("\nChanges detected. Proceed with overwrite?")
    ans = input("[Enter = Yes / n = No] : ").strip().lower()
    if ans == "n" or ans == "no":
        raise UserAbort("User cancelled.")


def _opts_from_args(args) -> SyncOptions:
    return SyncOptions(
        primary_only=bool(args.primary_only),
        skip_snapshots=bool(args.skip_snapshots),
        dry_run=bool(args.dry_run),
        assume_yes=bool(args.yes),
    )


def cmd_pull(args):
    s = plan_diff(args.root, args.dataset, args.remote)
    _print_diff(s)
    _confirm_or_abort(s, assume_yes=args.yes)
    execute_pull(args.root, args.dataset, args.remote, _opts_from_args(args))
    if not args.dry_run:
        print("Pull complete.")


def cmd_push(args):
    s = plan_diff(args.root, args.dataset, args.remote)
    _print_diff(s)
    _confirm_or_abort(s, assume_yes=args.yes)
    execute_push(args.root, args.dataset, args.remote, _opts_from_args(args))
    if not args.dry_run:
        print("Push complete.")
