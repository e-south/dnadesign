"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli_commands/write.py

USR CLI write/mutation command implementations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from ..dataset import Dataset


def cmd_init(args) -> None:
    dataset = Dataset(args.root, args.dataset)
    dataset.init(source=args.source, notes=args.notes)
    print(f"Initialized dataset at {dataset.records_path}")


def cmd_import(
    args,
    *,
    resolve_path_anywhere: Callable[[Path], Path],
) -> None:
    dataset = Dataset(args.root, args.dataset)
    in_path = resolve_path_anywhere(args.path)
    if args.source_format == "csv":
        df = pd.read_csv(in_path)
    else:
        df = pd.read_json(in_path, lines=True)
    count = dataset.import_rows(
        df,
        default_bio_type=args.bio_type,
        default_alphabet=args.alphabet,
        source=str(in_path),
    )
    print(f"Imported {count} records into {dataset.name}")
    cmd = (
        f"usr import {args.dataset} --from {args.source_format} --path {in_path} "
        f"--bio-type {args.bio_type} --alphabet {args.alphabet}"
    )
    dataset.append_meta_note(f"Imported {count} records from {in_path}", cmd)


def cmd_attach(
    args,
    *,
    resolve_path_anywhere: Callable[[Path], Path],
) -> None:
    dataset = Dataset(args.root, args.dataset)
    in_path = resolve_path_anywhere(args.path)
    columns = [c.strip() for c in args.columns.split(",")] if args.columns else None
    count = dataset.attach_columns(
        in_path,
        namespace=args.namespace,
        key=args.key,
        key_col=(args.key_col if args.key_col else None),
        columns=columns,
        allow_overwrite=bool(args.allow_overwrite),
        allow_missing=bool(args.allow_missing),
        parse_json=bool(args.parse_json),
        backend=args.backend,
        note=args.note,
    )
    msg = f"Attached {count} matched row(s) of {args.namespace} columns into {dataset.name} (input file: {in_path})"
    if args.allow_missing:
        msg += " (unmatched rows skipped; see .events.log for counts)"
    print(msg)

    cols = args.columns or "(all columns)"
    cmd = (
        f"usr attach {args.dataset} --path {in_path} --namespace {args.namespace} "
        f'--key {args.key} --key-col {args.key_col or ""} --columns "{cols}"'
    )
    if args.allow_overwrite:
        cmd += " --allow-overwrite"
    if args.allow_missing:
        cmd += " --allow-missing"
    if not args.parse_json:
        cmd += " --no-parse-json"
    if args.backend != "pyarrow":
        cmd += f" --backend {args.backend}"
    dataset.append_meta_note(f"Attached columns under '{args.namespace}' ({count} row match)", cmd)
