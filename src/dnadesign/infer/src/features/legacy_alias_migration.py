"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/features/legacy_alias_migration.py

Bridge verified legacy row-overlay feature values into sequence-view alias
sidecars without mutating the original Infer overlays.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from dnadesign.usr import Dataset

from ..contracts import infer_usr_column_name
from ..runtime.resume_planner import read_usr_columns
from .aliases import compute_feature_alias_id, persist_feature_alias_rows, persist_feature_vector_rows
from .contracts import SequenceContextRecord, SequenceFeatureBundleConfig
from .execution import (
    _sequence_view_feature_vector_specs,
    build_feature_metadata_rows,
)
from .selectors import resolve_intermediate_selector
from .sequence_views import load_sequence_view_input_records, resolve_sequence_view_contexts


@dataclass(frozen=True)
class LegacyAliasMigrationResult:
    model_id: str
    legacy_job_id: str
    required_views: int
    required_vectors: int
    reusable_vectors: int
    payload_unverified_vectors: int
    missing_vectors: int
    unclassified_vectors: int
    orientation_blocked_vectors: int
    vectors_written: int
    aliases_written: int
    by_product_kind: dict[str, int]
    by_orientation: dict[str, int]
    by_pooling_operation: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def migrate_legacy_overlay_aliases(
    *,
    bundle: SequenceFeatureBundleConfig,
    model_id: str,
    legacy_job_id: str,
    write: bool = False,
    assumed_legacy_orientation: str = "forward",
    max_views: int | None = None,
    verify_payloads: bool = False,
) -> LegacyAliasMigrationResult:
    """Plan or write alias/vector sidecars from verified legacy row overlays.

    The legacy row overlays remain untouched. This bridge is intentionally
    conservative because old overlays predate sequence-view ids and emitted
    orientation metadata.
    """

    records = load_sequence_view_input_records(bundle=bundle)
    if max_views is not None:
        if max_views <= 0:
            raise ValueError("max_views must be positive when provided.")
        records = records[:max_views]
    contexts = resolve_sequence_view_contexts(records=records)
    selector = resolve_intermediate_selector(model_id=model_id, intermediate_block=bundle.intermediate_block)
    metadata_rows = build_feature_metadata_rows(contexts=contexts, bundle=bundle, model_id=model_id)
    specs = _sequence_view_feature_vector_specs(
        contexts=contexts,
        metadata_rows=metadata_rows,
        bundle=bundle,
        selector=selector.intermediate_selector,
    )
    legacy_values = _read_legacy_overlay_values(
        contexts=contexts,
        specs=specs,
        model_id=model_id,
        legacy_job_id=legacy_job_id,
        include_feature_values=write or verify_payloads,
    )

    reusable_specs: list[dict[str, object]] = []
    reusable_values: list[list[float]] = []
    payload_unverified = 0
    missing = 0
    unclassified = 0
    orientation_blocked = 0
    for spec in specs:
        row_index = int(spec["row_index"])
        context = contexts[row_index]
        legacy_row = legacy_values.get((row_index, str(spec["out_id"])))
        if legacy_row is None or not _has_legacy_metadata(legacy_row):
            missing += 1
            continue
        if (write or verify_payloads) and legacy_row.get("value") is None:
            missing += 1
            continue
        if str(context.orientation or context.anchor_orientation or "forward") != assumed_legacy_orientation:
            orientation_blocked += 1
            continue
        if not _legacy_metadata_proves_identity(
            legacy_row=legacy_row,
            context=context,
            model_id=model_id,
            intermediate_selector=selector.intermediate_selector,
        ):
            unclassified += 1
            continue
        reusable_specs.append(spec)
        if write:
            reusable_values.append([float(item) for item in legacy_row["value"]])
        elif not verify_payloads:
            payload_unverified += 1

    vectors_written = 0
    aliases_written = 0
    if write and reusable_specs:
        vector_rows = [
            {
                "_dataset_root": spec["dataset_root"],
                "_dataset_id": spec["dataset_id"],
                "feature_vector_key": spec["feature_vector_key"],
                "value": value,
                "created_at": _legacy_created_at(
                    legacy_values.get((int(spec["row_index"]), str(spec["out_id"]))) or {},
                    fallback=str(metadata_rows[int(spec["row_index"])]["timestamp"]),
                ),
            }
            for spec, value in zip(reusable_specs, reusable_values, strict=True)
        ]
        alias_rows = [
            _alias_row_for_spec(
                spec=spec,
                context=contexts[int(spec["row_index"])],
                metadata=metadata_rows[int(spec["row_index"])],
                model_id=model_id,
                layer_name=(
                    selector.intermediate_selector if _representation_kind(spec) == "intermediate_embedding" else None
                ),
            )
            for spec in reusable_specs
        ]
        vectors_written = persist_feature_vector_rows(vector_rows)
        aliases_written = persist_feature_alias_rows(alias_rows)

    product_counts = Counter(str(context.product_kind) for context in contexts if context.product_kind is not None)
    orientation_counts = Counter(
        str(context.orientation or context.anchor_orientation or "forward") for context in contexts
    )
    pooling_counts = Counter(str(context.pooling_operation or "seq_mean") for context in contexts)
    return LegacyAliasMigrationResult(
        model_id=model_id,
        legacy_job_id=legacy_job_id,
        required_views=len(contexts),
        required_vectors=len(specs),
        reusable_vectors=len(reusable_specs),
        payload_unverified_vectors=payload_unverified,
        missing_vectors=missing,
        unclassified_vectors=unclassified,
        orientation_blocked_vectors=orientation_blocked,
        vectors_written=vectors_written,
        aliases_written=aliases_written,
        by_product_kind=dict(sorted(product_counts.items())),
        by_orientation=dict(sorted(orientation_counts.items())),
        by_pooling_operation=dict(sorted(pooling_counts.items())),
    )


def _read_legacy_overlay_values(
    *,
    contexts: list[SequenceContextRecord],
    specs: list[dict[str, object]],
    model_id: str,
    legacy_job_id: str,
    include_feature_values: bool,
) -> dict[tuple[int, str], dict[str, object]]:
    specs_by_dataset: dict[tuple[str, str], list[dict[str, object]]] = {}
    row_indexes_by_dataset: dict[tuple[str, str], set[int]] = {}
    for spec in specs:
        dataset_key = (str(spec["dataset_root"]), str(spec["dataset_id"]))
        specs_by_dataset.setdefault(dataset_key, []).append(spec)
        row_indexes_by_dataset.setdefault(dataset_key, set()).add(int(spec["row_index"]))

    values_by_spec: dict[tuple[int, str], dict[str, object]] = {}
    for (dataset_root, dataset_id), dataset_specs in specs_by_dataset.items():
        row_indexes = sorted(row_indexes_by_dataset[(dataset_root, dataset_id)])
        ids = [contexts[row_index].sequence_id for row_index in row_indexes]
        row_position = {row_index: position for position, row_index in enumerate(row_indexes)}
        output_columns: dict[str, str] = {}
        if include_feature_values:
            output_columns = {
                str(spec["out_id"]): infer_usr_column_name(
                    model_id=model_id,
                    job_id=legacy_job_id,
                    out_id=str(spec["out_id"]),
                )
                for spec in dataset_specs
            }
        metadata_columns = {
            name: infer_usr_column_name(model_id=model_id, job_id=legacy_job_id, out_id=name)
            for name in (
                "metadata__sequence_id",
                "metadata__context_kind",
                "metadata__resolved_length",
                "metadata__anchor_start",
                "metadata__anchor_end",
                "metadata__model_name",
                "metadata__intermediate_selector",
                "metadata__timestamp",
            )
        }
        columns = [*output_columns.values(), *metadata_columns.values()]
        column_values = read_usr_columns(
            ds=Dataset(Path(dataset_root), dataset_id),
            ids=ids,
            column_names=columns,
        )
        for spec in dataset_specs:
            row_index = int(spec["row_index"])
            position = row_position[row_index]
            out_id = str(spec["out_id"])
            values_by_spec[(row_index, out_id)] = {
                "value": (
                    column_values.get(output_columns[out_id], [None] * len(ids))[position]
                    if include_feature_values
                    else None
                ),
                **{
                    metadata_name: column_values.get(metadata_column, [None] * len(ids))[position]
                    for metadata_name, metadata_column in metadata_columns.items()
                },
            }
    return values_by_spec


def _has_legacy_metadata(row: dict[str, object]) -> bool:
    return bool(str(row.get("metadata__sequence_id") or "").strip())


def _legacy_metadata_proves_identity(
    *,
    legacy_row: dict[str, object],
    context: SequenceContextRecord,
    model_id: str,
    intermediate_selector: str,
) -> bool:
    if str(legacy_row.get("metadata__sequence_id") or "") != str(context.sequence_id):
        return False
    if str(legacy_row.get("metadata__model_name") or "") != str(model_id):
        return False
    if str(legacy_row.get("metadata__intermediate_selector") or "") != str(intermediate_selector):
        return False
    if str(legacy_row.get("metadata__context_kind") or "") != str(context.context_kind):
        return False
    if _int_or_none(legacy_row.get("metadata__resolved_length")) != int(context.resolved_length):
        return False
    if str(context.pooling_operation or "seq_mean") == "anchor_mean":
        if _int_or_none(legacy_row.get("metadata__anchor_start")) != int(context.anchor_start):
            return False
        if _int_or_none(legacy_row.get("metadata__anchor_end")) != int(context.anchor_end):
            return False
    return True


def _alias_row_for_spec(
    *,
    spec: dict[str, object],
    context: SequenceContextRecord,
    metadata: dict[str, object],
    model_id: str,
    layer_name: str | None,
) -> dict[str, object]:
    representation_kind = _representation_kind(spec)
    pooling_operation = _pooling_operation_from_vector_key(spec, context)
    pooling_start_0, pooling_end_0 = _pooling_bounds_for_alias(context, pooling_operation=pooling_operation)
    feature_vector_key = str(spec["feature_vector_key"])
    return {
        "_dataset_root": spec["dataset_root"],
        "_dataset_id": spec["dataset_id"],
        "alias_id": compute_feature_alias_id(
            view_id=context.view_id,
            sequence_id=context.sequence_id,
            feature_vector_key=feature_vector_key,
            representation_kind=representation_kind,
        ),
        "view_id": context.view_id,
        "view_name": context.view_name,
        "sequence_id": context.sequence_id,
        "feature_vector_key": feature_vector_key,
        "forward_pass_key": str(metadata["forward_pass_key"]),
        "provider": "evo2",
        "model_name": model_id,
        "model_revision": None,
        "layer_name": layer_name,
        "representation_kind": representation_kind,
        "pooling_operation": pooling_operation,
        "pooling_start_0": pooling_start_0,
        "pooling_end_0": pooling_end_0,
        "orientation": context.orientation or context.anchor_orientation or "forward",
        "source_dataset_id": context.source_dataset_id,
        "feature_request_digest": str(metadata["feature_request_digest"]),
        "created_at": str(metadata["timestamp"]),
    }


def _representation_kind(spec: dict[str, object]) -> str:
    out_id = str(spec["out_id"])
    if out_id.startswith("intermediate_embedding__"):
        return "intermediate_embedding"
    if out_id.startswith("output_layer_mean__"):
        return "output_layer_mean"
    raise ValueError(f"Unsupported sequence-view feature-vector output id for alias migration: {out_id}")


def _pooling_operation_from_vector_key(spec: dict[str, object], context: SequenceContextRecord) -> str:
    operation = str(context.pooling_operation or "seq_mean")
    if (
        operation == "core60_mean"
        and context.pooling_start_0 == 0
        and context.pooling_end_0 == context.resolved_length == 60
    ):
        return "seq_mean"
    return operation


def _pooling_bounds_for_alias(
    context: SequenceContextRecord,
    *,
    pooling_operation: str,
) -> tuple[int | None, int | None]:
    if pooling_operation == "seq_mean":
        return None, None
    return context.pooling_start_0, context.pooling_end_0


def _legacy_created_at(row: dict[str, object], *, fallback: str) -> str:
    text = str(row.get("metadata__timestamp") or "").strip()
    return text or fallback


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return int(text)
