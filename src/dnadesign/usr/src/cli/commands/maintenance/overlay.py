"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli/commands/maintenance/overlay.py

USR CLI maintenance overlay command implementations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ....contracts import SequencesError
from ....dataset import Dataset
from ....overlays.support.maintenance import remove_dataset_overlay
from ....overlays.support.projection import project_namespace_overlay
from .registry import MaintenanceDeps


def cmd_overlay_compact(args, *, deps: MaintenanceDeps) -> None:
    ds_name = deps.resolve_dataset_name_interactive(args.root, getattr(args, "dataset", None), False)
    if not ds_name:
        return
    namespace = getattr(args, "namespace", None)
    if not namespace:
        raise SequencesError("overlay-compact requires a namespace argument.")
    dataset = Dataset(args.root, ds_name)
    with dataset.maintenance(reason="overlay_compact"):
        out_path = dataset.compact_overlay(str(namespace))
    print(f"[overlay-compact] wrote {out_path}")


def cmd_overlay_refresh_metadata(args, *, deps: MaintenanceDeps) -> None:
    ds_name = deps.resolve_dataset_name_interactive(args.root, getattr(args, "dataset", None), False)
    if not ds_name:
        return
    namespace = getattr(args, "namespace", None)
    if not namespace:
        raise SequencesError("overlay-refresh-metadata requires a namespace argument.")
    dataset = Dataset(args.root, ds_name)
    with dataset.maintenance(reason="overlay_refresh_metadata"):
        result = dataset.refresh_overlay_metadata(str(namespace))
    print(
        "[overlay-refresh-metadata] "
        f"dataset={result.dataset} namespace={result.namespace} rows={result.rows_refreshed} "
        f"previous_registry_hash={result.previous_registry_hash} "
        f"refreshed_registry_hash={result.refreshed_registry_hash} "
        f"previous_namespace_contract_hash={result.previous_namespace_contract_hash} "
        f"refreshed_namespace_contract_hash={result.refreshed_namespace_contract_hash} "
        f"path={result.overlay_path}"
    )


def cmd_overlay_remove(args, *, deps: MaintenanceDeps) -> None:
    ds_name = deps.resolve_dataset_name_interactive(args.root, getattr(args, "dataset", None), False)
    if not ds_name:
        return
    namespace = getattr(args, "namespace", None)
    if not namespace:
        raise SequencesError("overlay-remove requires a namespace argument.")
    mode = str(getattr(args, "mode", "error") or "error")
    result = remove_dataset_overlay(args.root, ds_name, str(namespace), mode=mode)
    if result.get("removed"):
        archived_path = result.get("archived_path")
        if archived_path:
            print(f"[overlay-remove] removed namespace={result['namespace']} mode={mode} archived_path={archived_path}")
            return
        print(f"[overlay-remove] removed namespace={result['namespace']} mode={mode}")
        return
    print(f"[overlay-remove] no-op namespace={result['namespace']} mode={mode}")


def cmd_overlay_project(args, *, deps: MaintenanceDeps) -> None:
    del deps
    columns_text = str(getattr(args, "columns", "") or "")
    columns = [value.strip() for value in columns_text.split(",") if value.strip()] if columns_text else None
    preview = project_namespace_overlay(
        root=args.root,
        src_dataset_name=str(args.src),
        dest_dataset_name=str(args.dest),
        namespace=str(args.namespace),
        src_join=str(getattr(args, "src_join", "id") or "id"),
        dest_join=str(getattr(args, "dest_join", "id") or "id"),
        columns=columns,
        overwrite=bool(getattr(args, "overwrite", True)),
        allow_missing=bool(getattr(args, "allow_missing", False)),
        dry_run=bool(getattr(args, "dry_run", False)),
    )

    action = "DRY-RUN" if bool(getattr(args, "dry_run", False)) else "PROJECTED"
    cols = ",".join(preview.source_columns)
    print(
        f"[{action}] src='{preview.src_dataset}' dest='{preview.dest_dataset}' "
        f"namespace={preview.namespace} src_join={preview.src_join} dest_join={preview.dest_join} "
        f"columns={cols} matched={preview.matched_rows} missing={preview.missing_rows} dest_rows={preview.dest_rows}"
    )

    if bool(getattr(args, "dry_run", False)):
        return

    note = (
        f"usr maintenance overlay-project --src {preview.src_dataset} --dest {preview.dest_dataset} "
        f"--namespace {preview.namespace} --src-join {preview.src_join} --dest-join {preview.dest_join}"
    )
    if columns_text:
        note += f' --columns "{columns_text}"'
    if bool(getattr(args, "allow_missing", False)):
        note += " --allow-missing"
    if not bool(getattr(args, "overwrite", True)):
        note += " --no-overwrite"
    Dataset(args.root, preview.dest_dataset).append_meta_note(
        f"Projected namespace '{preview.namespace}' from '{preview.src_dataset}' ({preview.matched_rows} matched rows)",
        note,
    )
