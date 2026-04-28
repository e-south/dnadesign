"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/features/legacy_payload_retirement.py

Retire duplicated legacy row-overlay payload columns after sequence-view feature
vectors are protected in the canonical Infer sidecar store.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from dnadesign.usr import Dataset, overlay_digest_ledger_path, overlay_parts

from ..contracts import infer_usr_column_name
from .aliases import load_feature_vector_keys
from .contracts import PromoterFeatureBundleConfig
from .execution import _sequence_view_feature_vector_specs, build_feature_metadata_rows
from .legacy_alias_migration import (
    _has_legacy_metadata,
    _legacy_metadata_proves_identity,
    _read_legacy_overlay_values,
)
from .selectors import resolve_intermediate_selector
from .sequence_views import load_sequence_view_input_records, resolve_sequence_view_contexts


@dataclass(frozen=True)
class RetiredLegacyPayloadPart:
    path: str
    before_size_bytes: int
    after_size_bytes: int
    removed_columns: list[str]
    retained_columns: int
    deleted_file: bool


@dataclass(frozen=True)
class LegacyPayloadRetirementResult:
    model_id: str
    legacy_job_id: str
    required_vectors: int
    legacy_payload_vectors: int
    protected_vectors: int
    missing_modern_vectors: int
    unclassified_legacy_vectors: int
    orientation_blocked_vectors: int
    candidate_columns: list[str]
    legacy_parts_scanned: int
    legacy_parts_with_payload: int
    bytes_before: int
    bytes_after: int
    bytes_reclaimable: int
    bytes_reclaimed: int
    files_rewritten: int
    files_deleted: int
    mode: str
    by_product_kind: dict[str, int]
    by_orientation: dict[str, int]
    by_pooling_operation: dict[str, int]
    retired_parts: list[RetiredLegacyPayloadPart]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["retired_parts"] = [asdict(part) for part in self.retired_parts]
        return payload


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
    retired_parts: list[RetiredLegacyPayloadPart]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["retired_parts"] = [asdict(part) for part in self.retired_parts]
        return payload


def retire_legacy_overlay_payloads(
    *,
    bundle: PromoterFeatureBundleConfig,
    model_id: str,
    legacy_job_id: str,
    write: bool = False,
    delete_empty_parts: bool = True,
) -> LegacyPayloadRetirementResult:
    """Drop legacy payload columns only after canonical vector keys exist.

    This is a storage-retirement step, not a semantic migration. It refuses to
    mutate row-overlay files unless every requested sequence-view feature vector
    is already present in `_derived/infer/feature_vectors.parquet`.
    """

    records = load_sequence_view_input_records(bundle=bundle)
    contexts = resolve_sequence_view_contexts(records=records)
    selector = resolve_intermediate_selector(model_id=model_id, intermediate_block=bundle.intermediate_block)
    metadata_rows = build_feature_metadata_rows(contexts=contexts, bundle=bundle, model_id=model_id)
    specs = _sequence_view_feature_vector_specs(
        contexts=contexts,
        metadata_rows=metadata_rows,
        bundle=bundle,
        selector=selector.intermediate_selector,
    )
    legacy_identity = _classify_legacy_identity(
        contexts=contexts,
        specs=specs,
        model_id=model_id,
        legacy_job_id=legacy_job_id,
        intermediate_selector=selector.intermediate_selector,
    )
    required_keys_by_dataset = _required_keys_by_dataset(legacy_identity.reusable_specs)
    protected_keys_by_dataset = {
        dataset_key: load_feature_vector_keys(
            dataset_root=dataset_key[0],
            dataset_id=dataset_key[1],
            keys=keys,
        )
        for dataset_key, keys in required_keys_by_dataset.items()
    }
    protected_vectors = sum(len(keys) for keys in protected_keys_by_dataset.values())
    legacy_payload_vectors = sum(len(keys) for keys in required_keys_by_dataset.values())
    missing_modern_vectors = legacy_payload_vectors - protected_vectors
    if write and missing_modern_vectors:
        raise ValueError(
            "Refusing to retire legacy payload columns before canonical feature-vector protection is complete: "
            f"missing_modern_vectors={missing_modern_vectors}."
        )
    if write and (legacy_identity.unclassified_vectors or legacy_identity.orientation_blocked_vectors):
        raise ValueError(
            "Refusing to retire legacy payload columns with unproven legacy identity: "
            f"unclassified_vectors={legacy_identity.unclassified_vectors} "
            f"orientation_blocked_vectors={legacy_identity.orientation_blocked_vectors}."
        )

    candidate_columns = sorted(
        {
            infer_usr_column_name(
                model_id=model_id,
                job_id=legacy_job_id,
                out_id=str(spec["out_id"]),
            )
            for spec in specs
        }
    )
    retirement_plan = _plan_legacy_payload_retirement(
        specs=specs,
        candidate_columns=candidate_columns,
    )

    retired_parts: list[RetiredLegacyPayloadPart] = []
    if write:
        for part in retirement_plan.parts:
            retired_parts.append(
                _rewrite_or_delete_payload_part(
                    part,
                    delete_empty_parts=delete_empty_parts,
                )
            )
        _refresh_infer_overlay_ledgers(specs=specs)
        _log_retirement_events(
            specs=specs,
            candidate_columns=candidate_columns,
            retired_parts=retired_parts,
            model_id=model_id,
            legacy_job_id=legacy_job_id,
        )
    else:
        retired_parts = [
            RetiredLegacyPayloadPart(
                path=part.path.as_posix(),
                before_size_bytes=part.before_size_bytes,
                after_size_bytes=part.estimated_after_size_bytes,
                removed_columns=list(part.removed_columns),
                retained_columns=part.retained_columns,
                deleted_file=part.retained_columns <= 1 and delete_empty_parts,
            )
            for part in retirement_plan.parts
        ]

    product_counts = Counter(str(context.product_kind) for context in contexts if context.product_kind is not None)
    orientation_counts = Counter(
        str(context.orientation or context.anchor_orientation or "forward") for context in contexts
    )
    pooling_counts = Counter(str(context.pooling_operation or "seq_mean") for context in contexts)
    bytes_after = (
        sum(part.after_size_bytes for part in retired_parts) if write else retirement_plan.estimated_after_size_bytes
    )
    return LegacyPayloadRetirementResult(
        model_id=model_id,
        legacy_job_id=legacy_job_id,
        required_vectors=len(specs),
        legacy_payload_vectors=legacy_payload_vectors,
        protected_vectors=protected_vectors,
        missing_modern_vectors=missing_modern_vectors,
        unclassified_legacy_vectors=legacy_identity.unclassified_vectors,
        orientation_blocked_vectors=legacy_identity.orientation_blocked_vectors,
        candidate_columns=candidate_columns,
        legacy_parts_scanned=retirement_plan.parts_scanned,
        legacy_parts_with_payload=len(retirement_plan.parts),
        bytes_before=retirement_plan.bytes_before,
        bytes_after=bytes_after,
        bytes_reclaimable=max(0, retirement_plan.bytes_before - retirement_plan.estimated_after_size_bytes),
        bytes_reclaimed=max(0, retirement_plan.bytes_before - bytes_after) if write else 0,
        files_rewritten=sum(1 for part in retired_parts if not part.deleted_file) if write else 0,
        files_deleted=sum(1 for part in retired_parts if part.deleted_file) if write else 0,
        mode="write" if write else "dry_run",
        by_product_kind=dict(sorted(product_counts.items())),
        by_orientation=dict(sorted(orientation_counts.items())),
        by_pooling_operation=dict(sorted(pooling_counts.items())),
        retired_parts=retired_parts,
    )


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
    """Remove explicitly approved stale Infer overlay columns by metadata scan.

    This path is deliberately separate from protected duplicate-vector
    retirement. Use it only for columns that are no longer part of a supported
    semantic lane and therefore do not have a canonical feature-vector sidecar.
    """

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
    retired_parts: list[RetiredLegacyPayloadPart] = []
    if write:
        for part in plan.parts:
            retired_parts.append(
                _rewrite_or_delete_payload_part(
                    part,
                    delete_empty_parts=delete_empty_parts,
                )
            )
        if retired_parts:
            _refresh_or_remove_overlay_ledger(dataset=dataset, namespace=str(namespace))
            _log_stale_column_prune_event(
                dataset=dataset,
                namespace=str(namespace),
                column_prefixes=prefixes,
                column_names=names,
                reason=reason,
                retired_parts=retired_parts,
            )
    else:
        retired_parts = [
            RetiredLegacyPayloadPart(
                path=part.path.as_posix(),
                before_size_bytes=part.before_size_bytes,
                after_size_bytes=part.estimated_after_size_bytes,
                removed_columns=list(part.removed_columns),
                retained_columns=part.retained_columns,
                deleted_file=part.retained_columns <= 1 and delete_empty_parts,
            )
            for part in plan.parts
        ]

    removed_columns = sorted({column for part in retired_parts for column in part.removed_columns})
    bytes_after = sum(part.after_size_bytes for part in retired_parts) if write else plan.estimated_after_size_bytes
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
        files_rewritten=sum(1 for part in retired_parts if not part.deleted_file) if write else 0,
        files_deleted=sum(1 for part in retired_parts if part.deleted_file) if write else 0,
        mode="write" if write else "dry_run",
        retired_parts=retired_parts,
    )


@dataclass(frozen=True)
class _LegacyIdentityClassification:
    reusable_specs: list[dict[str, object]]
    unclassified_vectors: int
    orientation_blocked_vectors: int


def _classify_legacy_identity(
    *,
    contexts,
    specs: list[dict[str, object]],
    model_id: str,
    legacy_job_id: str,
    intermediate_selector: str,
    assumed_legacy_orientation: str = "forward",
) -> _LegacyIdentityClassification:
    legacy_values = _read_legacy_overlay_values(
        contexts=contexts,
        specs=specs,
        model_id=model_id,
        legacy_job_id=legacy_job_id,
        include_feature_values=False,
    )
    reusable_specs: list[dict[str, object]] = []
    unclassified = 0
    orientation_blocked = 0
    for spec in specs:
        row_index = int(spec["row_index"])
        context = contexts[row_index]
        legacy_row = legacy_values.get((row_index, str(spec["out_id"])))
        if legacy_row is None or not _has_legacy_metadata(legacy_row):
            continue
        if str(context.orientation or context.anchor_orientation or "forward") != assumed_legacy_orientation:
            orientation_blocked += 1
            continue
        if not _legacy_metadata_proves_identity(
            legacy_row=legacy_row,
            context=context,
            model_id=model_id,
            intermediate_selector=intermediate_selector,
        ):
            unclassified += 1
            continue
        reusable_specs.append(spec)
    return _LegacyIdentityClassification(
        reusable_specs=reusable_specs,
        unclassified_vectors=unclassified,
        orientation_blocked_vectors=orientation_blocked,
    )


@dataclass(frozen=True)
class _PayloadPartPlan:
    path: Path
    removed_columns: tuple[str, ...]
    retained_columns: int
    before_size_bytes: int
    estimated_after_size_bytes: int


@dataclass(frozen=True)
class _PayloadRetirementPlan:
    parts: list[_PayloadPartPlan]
    parts_scanned: int
    bytes_before: int
    estimated_after_size_bytes: int


def _required_keys_by_dataset(specs: list[dict[str, object]]) -> dict[tuple[str, str], set[str]]:
    out: dict[tuple[str, str], set[str]] = {}
    for spec in specs:
        dataset_key = (str(spec["dataset_root"]), str(spec["dataset_id"]))
        out.setdefault(dataset_key, set()).add(str(spec["feature_vector_key"]))
    return out


def _refresh_infer_overlay_ledgers(*, specs: list[dict[str, object]]) -> None:
    dataset_keys = sorted({(str(spec["dataset_root"]), str(spec["dataset_id"])) for spec in specs})
    for dataset_root, dataset_id in dataset_keys:
        _refresh_or_remove_overlay_ledger(dataset=Dataset(Path(dataset_root), dataset_id), namespace="infer")


def _refresh_or_remove_overlay_ledger(*, dataset: Dataset, namespace: str) -> None:
    overlay_dir = dataset.dir / "_derived" / namespace
    if overlay_parts(overlay_dir):
        dataset.write_overlay_digest_ledger(namespace)
        return
    ledger_path = overlay_digest_ledger_path(overlay_dir)
    if ledger_path is not None and ledger_path.exists():
        ledger_path.unlink()


def _plan_legacy_payload_retirement(
    *,
    specs: list[dict[str, object]],
    candidate_columns: list[str],
) -> _PayloadRetirementPlan:
    candidate_set = set(candidate_columns)
    dataset_keys = sorted({(str(spec["dataset_root"]), str(spec["dataset_id"])) for spec in specs})
    parts: list[_PayloadPartPlan] = []
    parts_scanned = 0
    for dataset_root, dataset_id in dataset_keys:
        for path in overlay_parts(Dataset(Path(dataset_root), dataset_id).dir / "_derived" / "infer"):
            parts_scanned += 1
            parquet_file = pq.ParquetFile(path)
            schema_names = parquet_file.schema_arrow.names
            removed = tuple(name for name in schema_names if name in candidate_set)
            if not removed:
                continue
            retained = [name for name in schema_names if name not in candidate_set]
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
    return _PayloadRetirementPlan(
        parts=parts,
        parts_scanned=parts_scanned,
        bytes_before=sum(part.before_size_bytes for part in parts),
        estimated_after_size_bytes=sum(part.estimated_after_size_bytes for part in parts),
    )


def _plan_stale_overlay_column_prune(
    *,
    dataset: Dataset,
    namespace: str,
    column_prefixes: tuple[str, ...],
    column_names: tuple[str, ...],
) -> _PayloadRetirementPlan:
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
    return _PayloadRetirementPlan(
        parts=parts,
        parts_scanned=parts_scanned,
        bytes_before=sum(part.before_size_bytes for part in parts),
        estimated_after_size_bytes=sum(part.estimated_after_size_bytes for part in parts),
    )


def _clean_selectors(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(value).strip() for value in values if str(value).strip())


def _estimate_retained_size(parquet_file: pq.ParquetFile, *, retained_columns: list[str]) -> int:
    """Estimate retained bytes from parquet column chunk metadata."""

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
) -> RetiredLegacyPayloadPart:
    path = plan.path
    parquet_file = pq.ParquetFile(path)
    original_schema = parquet_file.schema_arrow
    retained_columns = [name for name in original_schema.names if name not in set(plan.removed_columns)]
    before_size = path.stat().st_size
    if len(retained_columns) <= 1 and delete_empty_parts:
        path.unlink()
        return RetiredLegacyPayloadPart(
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
    return RetiredLegacyPayloadPart(
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


def _log_retirement_events(
    *,
    specs: list[dict[str, object]],
    candidate_columns: list[str],
    retired_parts: list[RetiredLegacyPayloadPart],
    model_id: str,
    legacy_job_id: str,
) -> None:
    dataset_keys = sorted({(str(spec["dataset_root"]), str(spec["dataset_id"])) for spec in specs})
    for dataset_root, dataset_id in dataset_keys:
        dataset = Dataset(Path(dataset_root), dataset_id)
        dataset_dir = dataset.dir.resolve()
        dataset_parts = [part for part in retired_parts if _is_relative_to(Path(part.path).resolve(), dataset_dir)]
        if not dataset_parts:
            continue
        dataset.log_event(
            "infer_legacy_payload_retirement",
            args={
                "model_id": model_id,
                "legacy_job_id": legacy_job_id,
                "retired_columns": candidate_columns,
            },
            metrics={
                "files_rewritten": sum(1 for part in dataset_parts if not part.deleted_file),
                "files_deleted": sum(1 for part in dataset_parts if part.deleted_file),
                "bytes_before": sum(part.before_size_bytes for part in dataset_parts),
                "bytes_after": sum(part.after_size_bytes for part in dataset_parts),
                "bytes_reclaimed": sum(
                    max(0, part.before_size_bytes - part.after_size_bytes) for part in dataset_parts
                ),
            },
            artifacts={
                "overlay_namespace": "infer",
                "parts": [Path(part.path).name for part in dataset_parts],
            },
            actor={"tool": "infer", "run_id": "legacy-payload-retirement"},
        )


def _log_stale_column_prune_event(
    *,
    dataset: Dataset,
    namespace: str,
    column_prefixes: tuple[str, ...],
    column_names: tuple[str, ...],
    reason: str,
    retired_parts: list[RetiredLegacyPayloadPart],
) -> None:
    removed_columns = sorted({column for part in retired_parts for column in part.removed_columns})
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
            "files_rewritten": sum(1 for part in retired_parts if not part.deleted_file),
            "files_deleted": sum(1 for part in retired_parts if part.deleted_file),
            "bytes_before": sum(part.before_size_bytes for part in retired_parts),
            "bytes_after": sum(part.after_size_bytes for part in retired_parts),
            "bytes_reclaimed": sum(max(0, part.before_size_bytes - part.after_size_bytes) for part in retired_parts),
            "columns_removed": len(removed_columns),
        },
        artifacts={
            "overlay_namespace": namespace,
            "parts": _compact_part_artifact_list(retired_parts),
            "parts_count": len(retired_parts),
        },
        actor={"tool": "infer", "run_id": "stale-overlay-column-prune"},
    )


def _compact_part_artifact_list(parts: list[RetiredLegacyPayloadPart], *, max_names: int = 20) -> list[str] | str:
    if len(parts) > max_names:
        return "omitted_large_part_list"
    return [Path(part.path).name for part in parts]


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False
