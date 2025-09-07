"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/usr/src/cli.py

USR CLI: thin wrapper around the Dataset API.

Default layout (editable install):
    src/dnadesign/usr/
      ├─ src/                  # package code (this file lives here)
      ├─ datasets/             # <-- default root for dataset folders
      │    └─ <dataset_name>/
      │         ├─ records.parquet
      │         ├─ meta.yaml            # optional; embedded metadata lives in records.parquet
      │         └─ _snapshots/          # bounded by module-level retention (see io.py)
      └─ template_demo/        # example CSVs for the README walkthrough

You can override the root with --root (or just cd into the datasets/ dir and use ".").

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq

from .dataset import Dataset
from .config import SSHRemoteConfig, load_all, save_remote, get_remote
from .sync import SyncOptions, plan_diff, execute_pull, execute_push
from .errors import UserAbort, SequencesError


def main() -> None:
    p = argparse.ArgumentParser(
        prog="usr",
        description="USR CLI (Parquet-backed datasets; includes SSH remotes sync).",
    )

    default_root = (Path(__file__).resolve().parents[1] / "datasets").resolve()

    p.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help=f"Datasets root folder (defaults to {default_root})",
    )
    sp = p.add_subparsers(dest="cmd", required=True)

    # ---------------- core dataset commands (unchanged) ----------------
    sp_ls = sp.add_parser("ls", help="List datasets under root")
    sp_ls.set_defaults(func=cmd_ls)

    sp_init = sp.add_parser("init", help="Create an empty dataset")
    sp_init.add_argument("dataset")
    sp_init.add_argument("--source", default="")
    sp_init.add_argument("--notes", default="")
    sp_init.set_defaults(func=cmd_init)

    sp_imp = sp.add_parser("import", help="Import sequences into a dataset")
    sp_imp.add_argument("dataset")
    sp_imp.add_argument("--from", dest="source_format", choices=["csv", "jsonl"], required=True)
    sp_imp.add_argument("--path", type=Path, required=True)
    sp_imp.add_argument("--bio-type", default="dna", choices=["dna", "protein"])
    sp_imp.add_argument("--alphabet", default="dna_4")
    sp_imp.set_defaults(func=cmd_import)

    sp_att = sp.add_parser("attach", help="Attach namespaced columns to a dataset")
    sp_att.add_argument("dataset")
    sp_att.add_argument("--path", type=Path, required=True)
    sp_att.add_argument("--namespace", required=True)
    sp_att.add_argument("--id-col", default="id")
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

    # ---------------- new: remotes management ----------------
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

    # ---------------- new: diff/status/pull/push ----------------
    def add_sync_common(p_: argparse.ArgumentParser):
        p_.add_argument("dataset")
        p_.add_argument("--remote", "--from", "--to", dest="remote", required=True)
        p_.add_argument("-y", "--yes", action="store_true", help="Assume yes (no prompt)")
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


# ---------- existing simple commands ----------

def list_datasets(root: Path):
    root = root.resolve()
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir() and (p / "records.parquet").exists()])


def cmd_ls(args):
    ds = list_datasets(args.root)
    if not ds:
        print(f"(no datasets under {args.root})"); return
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
    n = (
        d.import_csv(args.path, default_bio_type=args.bio_type, default_alphabet=args.alphabet)
        if args.source_format == "csv" else
        d.import_jsonl(args.path, default_bio_type=args.bio_type, default_alphabet=args.alphabet)
    )
    print(f"Imported {n} records into {d.name}")


def cmd_attach(args):
    d = Dataset(args.root, args.dataset)
    cols = [c.strip() for c in args.columns.split(",")] if args.columns else None
    n = d.attach(
        args.path, namespace=args.namespace, id_col=args.id_col, columns=cols,
        allow_overwrite=bool(args.allow_overwrite), note=args.note,
    )
    print(f"Attached {n} rows worth of {args.namespace} columns into {d.name}")


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
    print(f"type     : ssh")
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
        if n is None: return "?"
        for unit in ["B","KB","MB","GB","TB"]:
            if n < 1024: return f"{n:.0f}{unit}"
            n /= 1024
        return f"{n:.0f}PB"

    pl, pr = summary.primary_local, summary.primary_remote
    print(f"Dataset: {summary.dataset}")
    print(f"Local  : {pl.sha256 or '?'}  size={fmt_sz(pl.size)}  rows={pl.rows or '?'}  cols={pl.cols or '?'}")
    print(f"Remote : {pr.sha256 or '?'}  size={fmt_sz(pr.size)}  rows={pr.rows or '?'}  cols={pr.cols or '?'}")
    eq = "==" if (pl.sha256 and pr.sha256 and pl.sha256 == pr.sha256) else "≠"
    print(f"Primary sha: {pl.sha256 or '?'} {eq} {pr.sha256 or '?'}")
    print(f"meta.yaml   mtime: {summary.meta_local_mtime or '-'}  →  {summary.meta_remote_mtime or '-'}")
    delta_evt = max(0, summary.events_remote_lines - summary.events_local_lines)
    print(f".events.log lines: local={summary.events_local_lines}  remote={summary.events_remote_lines}  (+{delta_evt} on remote)")
    print(f"_snapshots : remote_count={summary.snapshots.count}  newer_than_local={summary.snapshots.newer_than_local}")
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
    res = execute_pull(args.root, args.dataset, args.remote, _opts_from_args(args))
    if not args.dry_run:
        print("Pull complete.")


def cmd_push(args):
    s = plan_diff(args.root, args.dataset, args.remote)
    _print_diff(s)
    _confirm_or_abort(s, assume_yes=args.yes)
    res = execute_push(args.root, args.dataset, args.remote, _opts_from_args(args))
    if not args.dry_run:
        print("Push complete.")
