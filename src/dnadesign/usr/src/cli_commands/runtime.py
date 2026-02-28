"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli_commands/runtime.py

Runtime command handlers for validation, events, state, and export flows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..cli_event_output import emit_event_line as emit_event_line_value
from ..dataset import Dataset
from ..errors import SequencesError
from ..pretty import PrettyOpts, fmt_value
from ..ui import print_df_plain, render_table_rich
from . import state as state_commands


@dataclass(frozen=True)
class RuntimeDeps:
    resolve_dataset_for_read: Callable[[Path, str], Dataset]
    resolve_dataset_name_interactive: Callable[[Path, str | None, bool], str | None]
    resolve_output_format: Callable[[object], str]
    print_json: Callable[[dict], None]
    output_version: int


def _emit_event_line(line: str, fmt: str) -> None:
    emitted = emit_event_line_value(line, fmt)
    if emitted is not None:
        print(emitted)


def cmd_validate(args, *, deps: RuntimeDeps) -> None:
    dataset_arg = getattr(args, "dataset", None)
    if dataset_arg:
        dataset = deps.resolve_dataset_for_read(args.root, str(dataset_arg))
    else:
        ds_name = deps.resolve_dataset_name_interactive(args.root, dataset_arg, False)
        if not ds_name:
            return
        dataset = Dataset(args.root, ds_name)
    dataset.validate(
        strict=bool(getattr(args, "strict", False)),
        registry_mode=str(getattr(args, "registry_mode", "current")),
    )
    print("OK: validation passed.")


def cmd_events_tail(args, *, deps: RuntimeDeps) -> None:
    dataset_arg = getattr(args, "dataset", None)
    if dataset_arg:
        dataset = deps.resolve_dataset_for_read(args.root, str(dataset_arg))
    else:
        ds_name = deps.resolve_dataset_name_interactive(args.root, dataset_arg, False)
        if not ds_name:
            return
        dataset = Dataset(args.root, ds_name)
    events_path = dataset.events_path
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

    with events_path.open("r", encoding="utf-8") as handle:
        handle.seek(0, os.SEEK_END)
        while True:
            line = handle.readline()
            if line:
                _emit_event_line(line, fmt)
                continue
            time.sleep(0.2)


def cmd_get(args, *, deps: RuntimeDeps) -> None:
    ds_name = deps.resolve_dataset_name_interactive(
        args.root, getattr(args, "dataset", None), bool(getattr(args, "rich", False))
    )
    if not ds_name:
        return
    rid = getattr(args, "id", None) or getattr(args, "id_positional", None)
    if not rid:
        print("Usage: usr get [dataset] --id <sha1>  (or)  usr get <sha1>")
        return
    dataset = Dataset(args.root, ds_name)
    cols = [c.strip() for c in args.columns.split(",")] if args.columns else None
    df = dataset.get(rid, columns=cols, include_deleted=bool(getattr(args, "include_deleted", False)))
    if df.empty:
        print("Not found.")
    elif getattr(args, "rich", False):
        df_fmt = df.applymap(lambda x: fmt_value(x, PrettyOpts()))
        render_table_rich(df_fmt, title=f"record: {rid}", caption=str(dataset.records_path))
    else:
        print_df_plain(df)


def cmd_grep(args, *, deps: RuntimeDeps) -> None:
    ds_name = deps.resolve_dataset_name_interactive(
        args.root, getattr(args, "dataset", None), bool(getattr(args, "rich", False))
    )
    if not ds_name:
        return
    dataset = Dataset(args.root, ds_name)
    df = dataset.grep(
        args.pattern,
        args.limit,
        batch_size=int(args.batch_size),
        include_deleted=bool(getattr(args, "include_deleted", False)),
    )
    if getattr(args, "rich", False):
        df_fmt = df.applymap(lambda x: fmt_value(x, PrettyOpts()))
        render_table_rich(df_fmt, title=f"grep: {args.pattern}")
    else:
        print_df_plain(df)


def _default_export_filename(dataset_name: str, fmt: str) -> str:
    stem = Path(dataset_name).as_posix().strip("/").replace("/", "_")
    return f"{stem}.{fmt}"


def _resolve_export_target(out_path: Path, *, dataset_name: str, fmt: str) -> Path:
    target = Path(out_path)
    if target.exists() and target.is_dir():
        return target / _default_export_filename(dataset_name, fmt)
    return target


def cmd_export(args, *, deps: RuntimeDeps) -> None:
    dataset_arg = getattr(args, "dataset", None)
    if dataset_arg:
        dataset = deps.resolve_dataset_for_read(args.root, str(dataset_arg))
    else:
        ds_name = deps.resolve_dataset_name_interactive(args.root, dataset_arg, False)
        if not ds_name:
            return
        dataset = Dataset(args.root, ds_name)
    fmt = str(args.fmt or "").strip().lower()
    out_target = _resolve_export_target(Path(args.out), dataset_name=dataset.name, fmt=fmt)
    cols = [c.strip() for c in args.columns.split(",") if c.strip()] if args.columns else None
    dataset.export(fmt, out_target, columns=cols, include_deleted=bool(getattr(args, "include_deleted", False)))
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


def cmd_delete(args, *, deps: RuntimeDeps) -> None:
    _ = deps
    state_commands.cmd_delete(
        args,
        collect_ids=_collect_ids,
    )


def cmd_restore(args, *, deps: RuntimeDeps) -> None:
    _ = deps
    state_commands.cmd_restore(
        args,
        collect_ids=_collect_ids,
    )


def cmd_state_set(args, *, deps: RuntimeDeps) -> None:
    _ = deps
    state_commands.cmd_state_set(
        args,
        collect_ids=_collect_ids,
        collect_list=_collect_list,
    )


def cmd_state_clear(args, *, deps: RuntimeDeps) -> None:
    _ = deps
    state_commands.cmd_state_clear(
        args,
        collect_ids=_collect_ids,
    )


def cmd_state_get(args, *, deps: RuntimeDeps) -> None:
    state_commands.cmd_state_get(
        args,
        collect_ids=_collect_ids,
        resolve_output_format=deps.resolve_output_format,
        print_json=deps.print_json,
        output_version=deps.output_version,
    )
