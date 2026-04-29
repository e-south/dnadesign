"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/features/overlay_pruning.py

Schema-only pruning for stale Infer row-overlay columns.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.usr import Dataset, overlay_digest_ledger_path, overlay_parts


@dataclass(frozen=True)
class PrunedOverlayPart:
    path: str
    before_size_bytes: int
    after_size_bytes: int
    removed_columns: list[str]
    retained_columns: int
    deleted_file: bool


@dataclass(frozen=True)
class StaleInferOverlayColumnPruneResult:
    dataset_root: str
    dataset_id: str
    namespace: str
    column_prefixes: list[str]
    column_names: list[str]
    reason: str
    removed_columns: list[str]
    parts_scanned: int
    parts_with_columns: int
    bytes_before: int
    bytes_after: int
    bytes_reclaimable: int
    bytes_reclaimed: int
    files_rewritten: int
    files_deleted: int
    mode: str
    pruned_parts: list[PrunedOverlayPart]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["pruned_parts"] = [asdict(part) for part in self.pruned_parts]
        return payload


def prune_stale_infer_overlay_columns(
    *,
    dataset_root: str | Path,
    dataset_id: str,
    namespace: str = "infer",
    column_prefixes: Sequence[str] = (),
    column_names: Sequence[str] = (),
    reason: str = "",
    write: bool = False,
    delete_empty_parts: bool = True,
) -> StaleInferOverlayColumnPruneResult:
    """Remove explicitly approved stale Infer overlay columns by schema scan."""

    if str(namespace) != "infer":
        raise ValueError("stale Infer overlay pruning only supports namespace='infer'.")
    prefixes = _clean_selectors(column_prefixes)
    names = _clean_selectors(column_names)
    if not prefixes and not names:
        raise ValueError("At least one --column-prefix or --column-name selector is required.")
    if "id" in set(names):
        raise ValueError("Refusing to prune the required overlay join column 'id'.")

    dataset = Dataset(Path(dataset_root), str(dataset_id))
    plan = _plan_stale_overlay_column_prune(
        dataset=dataset,
        namespace=str(namespace),
        column_prefixes=prefixes,
        column_names=names,
    )
    pruned_parts: list[PrunedOverlayPart] = []
    if write:
        for part in plan.parts:
            pruned_parts.append(
                _rewrite_or_delete_payload_part(
                    part,
                    delete_empty_parts=delete_empty_parts,
                )
            )
        if pruned_parts:
            _refresh_or_remove_overlay_ledger(dataset=dataset, namespace=str(namespace))
            _log_stale_column_prune_event(
                dataset=dataset,
                namespace=str(namespace),
                column_prefixes=prefixes,
                column_names=names,
                reason=reason,
                pruned_parts=pruned_parts,
            )
    else:
        pruned_parts = [
            PrunedOverlayPart(
                path=part.path.as_posix(),
                before_size_bytes=part.before_size_bytes,
                after_size_bytes=part.estimated_after_size_bytes,
                removed_columns=list(part.removed_columns),
                retained_columns=part.retained_columns,
                deleted_file=part.retained_columns <= 1 and delete_empty_parts,
            )
            for part in plan.parts
        ]

    removed_columns = sorted({column for part in pruned_parts for column in part.removed_columns})
    bytes_after = sum(part.after_size_bytes for part in pruned_parts) if write else plan.estimated_after_size_bytes
    return StaleInferOverlayColumnPruneResult(
        dataset_root=str(Path(dataset_root)),
        dataset_id=str(dataset_id),
        namespace=str(namespace),
        column_prefixes=list(prefixes),
        column_names=list(names),
        reason=str(reason),
        removed_columns=removed_columns,
        parts_scanned=plan.parts_scanned,
        parts_with_columns=len(plan.parts),
        bytes_before=plan.bytes_before,
        bytes_after=bytes_after,
        bytes_reclaimable=max(0, plan.bytes_before - plan.estimated_after_size_bytes),
        bytes_reclaimed=max(0, plan.bytes_before - bytes_after) if write else 0,
        files_rewritten=sum(1 for part in pruned_parts if not part.deleted_file) if write else 0,
        files_deleted=sum(1 for part in pruned_parts if part.deleted_file) if write else 0,
        mode="write" if write else "dry_run",
        pruned_parts=pruned_parts,
    )


@dataclass(frozen=True)
class _PayloadPartPlan:
    path: Path
    removed_columns: tuple[str, ...]
    retained_columns: int
    before_size_bytes: int
    estimated_after_size_bytes: int


@dataclass(frozen=True)
class _PayloadPrunePlan:
    parts: list[_PayloadPartPlan]
    parts_scanned: int
    bytes_before: int
    estimated_after_size_bytes: int


def _plan_stale_overlay_column_prune(
    *,
    dataset: Dataset,
    namespace: str,
    column_prefixes: tuple[str, ...],
    column_names: tuple[str, ...],
) -> _PayloadPrunePlan:
    parts: list[_PayloadPartPlan] = []
    parts_scanned = 0
    overlay_dir = dataset.dir / "_derived" / namespace
    selected_names = set(column_names)
    for path in overlay_parts(overlay_dir):
        parts_scanned += 1
        parquet_file = pq.ParquetFile(path)
        schema_names = parquet_file.schema_arrow.names
        removed = tuple(
            name
            for name in schema_names
            if name in selected_names or any(name.startswith(prefix) for prefix in column_prefixes)
        )
        if "id" in removed:
            raise ValueError(f"Refusing to prune required overlay join column 'id' in {path}.")
        if not removed:
            continue
        retained = [name for name in schema_names if name not in set(removed)]
        before_size = path.stat().st_size
        parts.append(
            _PayloadPartPlan(
                path=path,
                removed_columns=removed,
                retained_columns=len(retained),
                before_size_bytes=before_size,
                estimated_after_size_bytes=_estimate_retained_size(parquet_file, retained_columns=retained),
            )
        )
    return _PayloadPrunePlan(
        parts=parts,
        parts_scanned=parts_scanned,
        bytes_before=sum(part.before_size_bytes for part in parts),
        estimated_after_size_bytes=sum(part.estimated_after_size_bytes for part in parts),
    )


def _clean_selectors(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(value).strip() for value in values if str(value).strip())


def _estimate_retained_size(parquet_file: pq.ParquetFile, *, retained_columns: list[str]) -> int:
    if not retained_columns:
        return 0
    retained = set(retained_columns)
    total = 0
    schema_names = parquet_file.schema_arrow.names
    for row_group_index in range(parquet_file.metadata.num_row_groups):
        row_group = parquet_file.metadata.row_group(row_group_index)
        for column_index, name in enumerate(schema_names):
            if name in retained:
                total += int(row_group.column(column_index).total_compressed_size)
    return total


def _rewrite_or_delete_payload_part(
    plan: _PayloadPartPlan,
    *,
    delete_empty_parts: bool,
) -> PrunedOverlayPart:
    path = plan.path
    parquet_file = pq.ParquetFile(path)
    original_schema = parquet_file.schema_arrow
    retained_columns = [name for name in original_schema.names if name not in set(plan.removed_columns)]
    before_size = path.stat().st_size
    if len(retained_columns) <= 1 and delete_empty_parts:
        path.unlink()
        return PrunedOverlayPart(
            path=path.as_posix(),
            before_size_bytes=before_size,
            after_size_bytes=0,
            removed_columns=list(plan.removed_columns),
            retained_columns=len(retained_columns),
            deleted_file=True,
        )

    retained_schema = _schema_with_preserved_metadata(original_schema, retained_columns=retained_columns)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}.", suffix=".parquet", delete=False) as handle:
        tmp_path = Path(handle.name)
    writer: pq.ParquetWriter | None = None
    try:
        writer = pq.ParquetWriter(tmp_path, retained_schema)
        for batch in parquet_file.iter_batches(batch_size=2048, columns=retained_columns):
            table = pa.Table.from_batches([batch]).cast(retained_schema)
            writer.write_table(table)
        writer.close()
        writer = None
        tmp_path.replace(path)
    finally:
        if writer is not None:
            writer.close()
        if tmp_path.exists():
            tmp_path.unlink()
    return PrunedOverlayPart(
        path=path.as_posix(),
        before_size_bytes=before_size,
        after_size_bytes=path.stat().st_size,
        removed_columns=list(plan.removed_columns),
        retained_columns=len(retained_columns),
        deleted_file=False,
    )


def _schema_with_preserved_metadata(schema: pa.Schema, *, retained_columns: list[str]) -> pa.Schema:
    fields = [schema.field(name) for name in retained_columns]
    return pa.schema(fields, metadata=schema.metadata)


def _refresh_or_remove_overlay_ledger(*, dataset: Dataset, namespace: str) -> None:
    overlay_dir = dataset.dir / "_derived" / namespace
    if overlay_parts(overlay_dir):
        dataset.write_overlay_digest_ledger(namespace)
        return
    ledger_path = overlay_digest_ledger_path(overlay_dir)
    if ledger_path is not None and ledger_path.exists():
        ledger_path.unlink()


def _log_stale_column_prune_event(
    *,
    dataset: Dataset,
    namespace: str,
    column_prefixes: tuple[str, ...],
    column_names: tuple[str, ...],
    reason: str,
    pruned_parts: list[PrunedOverlayPart],
) -> None:
    removed_columns = sorted({column for part in pruned_parts for column in part.removed_columns})
    dataset.log_event(
        "infer_stale_overlay_column_prune",
        args={
            "namespace": namespace,
            "column_prefixes": list(column_prefixes),
            "column_names": list(column_names),
            "reason": str(reason),
            "removed_columns": removed_columns,
        },
        metrics={
            "files_rewritten": sum(1 for part in pruned_parts if not part.deleted_file),
            "files_deleted": sum(1 for part in pruned_parts if part.deleted_file),
            "bytes_before": sum(part.before_size_bytes for part in pruned_parts),
            "bytes_after": sum(part.after_size_bytes for part in pruned_parts),
            "bytes_reclaimed": sum(max(0, part.before_size_bytes - part.after_size_bytes) for part in pruned_parts),
            "columns_removed": len(removed_columns),
        },
        artifacts={
            "overlay_namespace": namespace,
            "parts": _compact_part_artifact_list(pruned_parts),
            "parts_count": len(pruned_parts),
        },
        actor={"tool": "infer", "run_id": "stale-overlay-column-prune"},
    )


def _compact_part_artifact_list(parts: list[PrunedOverlayPart], *, max_names: int = 20) -> list[str] | str:
    if len(parts) > max_names:
        return "omitted_large_part_list"
    return [Path(part.path).name for part in parts]
