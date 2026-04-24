"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli/commands/lifecycle/state.py

USR CLI lifecycle state and soft-delete command implementations.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ....contracts import SequencesError
from ....dataset import Dataset
from ...support.presentation.rendering import print_df_plain, render_table_rich
from ..datasets import resolve_dataset_name_interactive


def _collect_ids(ids: list[str] | None, id_file: Path | None) -> list[str]:
    out: list[str] = []
    if ids:
        for value in ids:
            out.extend([token.strip() for token in str(value).split(",") if token.strip()])
    if id_file:
        text = Path(id_file).read_text(encoding="utf-8")
        out.extend([line.strip() for line in text.splitlines() if line.strip()])
    if not out:
        raise SequencesError("Provide at least one id via --id or --id-file.")
    return out


def _collect_list(vals: list[str] | None) -> list[str]:
    out: list[str] = []
    if vals:
        for value in vals:
            out.extend([token.strip() for token in str(value).split(",") if token.strip()])
    return out


def cmd_delete(args) -> None:
    ds_name = resolve_dataset_name_interactive(args.root, getattr(args, "dataset", None), False)
    if not ds_name:
        return
    dataset = Dataset(args.root, ds_name)
    ids = _collect_ids(getattr(args, "id", None), getattr(args, "id_file", None))
    count = dataset.tombstone(
        ids,
        reason=getattr(args, "reason", None),
        allow_missing=bool(getattr(args, "allow_missing", False)),
    )
    print(f"Tombstoned {count} record(s) in {dataset.name}")


def cmd_restore(args) -> None:
    ds_name = resolve_dataset_name_interactive(args.root, getattr(args, "dataset", None), False)
    if not ds_name:
        return
    dataset = Dataset(args.root, ds_name)
    ids = _collect_ids(getattr(args, "id", None), getattr(args, "id_file", None))
    count = dataset.restore(ids, allow_missing=bool(getattr(args, "allow_missing", False)))
    print(f"Restored {count} record(s) in {dataset.name}")


def cmd_state_set(args) -> None:
    ds_name = resolve_dataset_name_interactive(args.root, getattr(args, "dataset", None), False)
    if not ds_name:
        return
    dataset = Dataset(args.root, ds_name)
    ids = _collect_ids(getattr(args, "id", None), getattr(args, "id_file", None))
    lineage = _collect_list(getattr(args, "lineage", None))
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


def cmd_state_clear(args) -> None:
    ds_name = resolve_dataset_name_interactive(args.root, getattr(args, "dataset", None), False)
    if not ds_name:
        return
    dataset = Dataset(args.root, ds_name)
    ids = _collect_ids(getattr(args, "id", None), getattr(args, "id_file", None))
    rows = dataset.clear_state(ids, allow_missing=bool(getattr(args, "allow_missing", False)))
    print(f"Cleared usr_state for {rows} record(s) in {dataset.name}")


def cmd_state_get(
    args,
    *,
    resolve_output_format: Callable[[object], str],
    print_json: Callable[[dict], None],
    output_version: int,
) -> None:
    ds_name = resolve_dataset_name_interactive(args.root, getattr(args, "dataset", None), False)
    if not ds_name:
        return
    dataset = Dataset(args.root, ds_name)
    ids = _collect_ids(getattr(args, "id", None), getattr(args, "id_file", None))
    df = dataset.get_state(ids, allow_missing=bool(getattr(args, "allow_missing", False)))
    fmt = resolve_output_format(args)
    if fmt == "json":
        print_json({"usr_output_version": output_version, "data": df.to_dict(orient="records")})
        return
    if fmt == "rich":
        render_table_rich(df, title=f"usr_state: {dataset.name}")
        return
    print_df_plain(df)
