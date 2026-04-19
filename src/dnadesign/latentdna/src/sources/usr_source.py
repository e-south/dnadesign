"""
USR source helpers.
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.usr import Dataset
from dnadesign.usr.dataset import RESERVED_NAMESPACES
from dnadesign.usr.dataset_overlay_catalog import load_overlay_catalog
from dnadesign.usr.overlay_digest_ledger import (
    overlay_digest_ledger_path,
    write_overlay_digest_ledger,
)
from dnadesign.usr.overlays import overlay_parts

from .provenance import OVERLAY_INVENTORY_DIGEST_MODE


def records_path(root: str, dataset: str, *, workspace_dir: Path) -> Path:
    root_path = Path(root)
    if not root_path.is_absolute():
        root_path = workspace_dir / root_path
    return (root_path / dataset / "records.parquet").resolve()


def load_dataset(root: str, dataset: str, *, workspace_dir: Path) -> Dataset:
    root_path = Path(root)
    if not root_path.is_absolute():
        root_path = workspace_dir / root_path
    return Dataset(root_path.resolve(), dataset)


def source_provenance(
    root: str,
    dataset: str,
    *,
    workspace_dir: Path,
    columns: list[str] | None = None,
) -> list[dict[str, object]]:
    resolved_records = records_path(root, dataset, workspace_dir=workspace_dir)
    resolved_dataset = load_dataset(root, dataset, workspace_dir=workspace_dir)
    requested = set(columns) if columns is not None else None
    entries: list[dict[str, object]] = [
        {
            "kind": "file",
            "id": resolved_records.name,
            "path": resolved_records.as_posix(),
            "role": "records",
        }
    ]
    overlays = load_overlay_catalog(
        resolved_dataset,
        include_tombstone=False,
        reserved_namespaces=RESERVED_NAMESPACES,
    )
    for overlay in overlays:
        overlay_columns = [column for column in overlay["cols"] if requested is None or column in requested]
        if not overlay_columns:
            continue
        path = Path(str(overlay["path"]))
        namespace = str(overlay["namespace"])
        entries.append(
            {
                "kind": "directory" if path.is_dir() else "file",
                "id": namespace,
                "path": path.as_posix(),
                "role": "overlay",
                "namespace": namespace,
                "columns": overlay_columns,
                "digest_mode": OVERLAY_INVENTORY_DIGEST_MODE,
            }
        )
        ledger_path = overlay_digest_ledger_path(path)
        if ledger_path is not None and ledger_path.is_file():
            refreshed_ledger_path = write_overlay_digest_ledger(path)
            entries.append(
                {
                    "kind": "file",
                    "id": f"{namespace}:digest_ledger",
                    "path": refreshed_ledger_path.as_posix(),
                    "role": "overlay_ledger",
                    "namespace": namespace,
                    "columns": overlay_columns,
                }
            )
            continue
        for part in overlay_parts(path):
            part_path = Path(part)
            part_id = namespace if part_path == path else f"{namespace}:{part_path.name}"
            entries.append(
                {
                    "kind": "file",
                    "id": part_id,
                    "path": part_path.as_posix(),
                    "role": "overlay_part",
                    "namespace": namespace,
                    "columns": overlay_columns,
                }
            )
    return entries
