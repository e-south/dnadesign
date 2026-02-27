"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli_commands/materialize.py

Materialization command workflow for USR datasets.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..dataset import Dataset
from ..errors import SequencesError, UserAbort
from ..overlays import list_overlays, overlay_metadata


@dataclass(frozen=True)
class MaterializeDeps:
    resolve_dataset_name_interactive: Callable[[Path, str | None, bool], str | None]
    is_interactive: Callable[[], bool]
    confirm: Callable[[str], bool]


def cmd_materialize(args, *, deps: MaterializeDeps) -> None:
    ds_name = deps.resolve_dataset_name_interactive(args.root, getattr(args, "dataset", None), False)
    if not ds_name:
        return
    dataset = Dataset(args.root, ds_name)
    ns_filter = [s.strip() for s in (args.namespaces or "").split(",") if s.strip()]
    overlays = list_overlays(dataset.dir)
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
    if deps.is_interactive() and not getattr(args, "yes", False):
        overlay_names = []
        for path in overlays:
            meta = overlay_metadata(path)
            ns = meta.get("namespace") or path.stem
            overlay_names.append(ns)
        ns_list = ", ".join(sorted(set(overlay_names)))
        print("WARNING: materialize will rewrite records.parquet by merging overlays into the base table.")
        print(f"Dataset: {dataset.name}")
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
        if not deps.confirm("Proceed?"):
            raise UserAbort()
    if getattr(args, "snapshot_before", False):
        dataset.snapshot()
    with dataset.maintenance(reason="materialize"):
        dataset.materialize(
            namespaces=ns_filter or None,
            keep_overlays=not bool(getattr(args, "drop_overlays", False)),
            archive_overlays=bool(getattr(args, "archive_overlays", False)),
            drop_deleted=bool(getattr(args, "drop_deleted", False)),
        )
    print(f"Materialized {len(overlays)} overlay(s) into {dataset.records_path}")
    dataset.append_meta_note("Materialized overlays", f"usr materialize {ds_name}")
