"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/features/sequence_views.py

Sequence-view ingest helpers for explicit Infer view execution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from dnadesign.usr import (
    Dataset,
    normalize_usr_root,
    resolve_usr_root_from_env,
    sequence_views_path,
)

from ..errors import CapabilityError, ValidationError
from .context import _bool_or_none
from .contracts import SequenceContextRecord, SequenceFeatureBundleConfig


@dataclass(frozen=True)
class SequenceViewMissingProduct:
    dataset: str
    root: str
    product_kind: str | None
    view_name: str | None
    alias: str | None
    orientation: str | None
    pooling_operation: str

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "root": self.root,
            "product_kind": self.product_kind,
            "view_name": self.view_name,
            "alias": self.alias,
            "orientation": self.orientation,
            "pooling_operation": self.pooling_operation,
        }


@dataclass(frozen=True)
class SequenceViewInputLoadResult:
    records: list[dict[str, object]]
    missing_products: list[SequenceViewMissingProduct]


def bundle_uses_sequence_views(bundle: SequenceFeatureBundleConfig) -> bool:
    return bool(bundle.sequence_view_inputs)


def _resolve_usr_root(value: str | None) -> Path:
    if value is not None and str(value).strip():
        return normalize_usr_root(value)
    resolved = resolve_usr_root_from_env()
    if resolved is None:
        raise ValidationError("Sequence-view feature bundles require input.root or DNADESIGN_USR_ROOT.")
    return resolved


def _read_sequence_rows(ds: Dataset, *, ids: list[str]) -> dict[str, str]:
    if not ids:
        return {}
    table = pq.read_table(ds.records_path, columns=["id", "sequence"], filters=[("id", "in", ids)])
    return {
        str(row_id): str(sequence)
        for row_id, sequence in zip(table.column("id").to_pylist(), table.column("sequence").to_pylist(), strict=True)
    }


def _read_construct_rows(ds: Dataset, *, ids: list[str]) -> dict[str, dict[str, object]]:
    if not ids:
        return {}
    if hasattr(ds, "_duckdb_query"):
        columns = [
            "id",
            "construct__anchor_start",
            "construct__anchor_end",
            "construct__anchor_id",
            "construct__input_id",
            "construct__anchor_orientation",
            "construct__resolved_length",
            "construct__spec_id",
            "construct__template_id",
            "is_wildtype",
        ]
        placeholders = ", ".join("?" for _ in ids)
        con, query, params = ds._duckdb_query(
            columns=columns,
            include_overlays=True,
            include_deleted=False,
            where=f"b.id IN ({placeholders})",
            params=ids,
            limit=len(ids),
        )
        try:
            con.execute(query, params)
            rows: dict[str, dict[str, object]] = {}
            for batch in con.to_arrow_reader(max(len(ids), 1)):
                payload = batch.to_pydict()
                for row_index in range(batch.num_rows):
                    row = {name: payload[name][row_index] for name in payload}
                    rows[str(row["id"])] = row
            return rows
        finally:
            con.close()
    return {}


_SEQUENCE_VIEW_INPUT_COLUMNS = [
    "view_id",
    "sequence_id",
    "view_name",
    "aliases",
    "product_kind",
    "context_kind",
    "orientation",
    "parent_sequence_id",
    "derivation_id",
    "anchor_start_0",
    "anchor_end_0",
]


def _aliases_contain(aliases: object, value: str) -> bool:
    if not isinstance(aliases, list):
        return False
    return value.casefold() in {str(alias).casefold() for alias in aliases}


def _select_sequence_view_rows(
    ds: Dataset,
    *,
    product_kind: str | None,
    view_name: str | None,
    alias: str | None,
    orientation: str | None,
) -> list[dict[str, object]]:
    path = sequence_views_path(ds)
    if not path.exists():
        return []
    table = pq.read_table(path, columns=_SEQUENCE_VIEW_INPUT_COLUMNS)
    rows: list[dict[str, object]] = []
    for raw in table.to_pylist():
        if product_kind is not None and raw.get("product_kind") != product_kind:
            continue
        if view_name is not None and raw.get("view_name") != view_name:
            continue
        if alias is not None and not _aliases_contain(raw.get("aliases"), alias):
            continue
        if orientation is not None and raw.get("orientation") != orientation:
            continue
        rows.append(dict(raw))
    return rows


def _context_kind_from_product_kind(product_kind: str, fallback: str | None) -> str:
    if fallback:
        return fallback
    if product_kind == "construct_insert":
        return "anchor_only"
    if product_kind == "analysis_window":
        return "analysis_window"
    if product_kind == "realized_context":
        return "template_1kb"
    return "native_reference"


def _pooling_bounds_from_record(
    *,
    sequence: str,
    record: dict[str, object],
) -> tuple[int | None, int | None]:
    operation = str(record["_infer_pooling_operation"])
    if operation == "seq_mean":
        return None, None
    if operation == "core60_mean":
        start = record.get("_infer_pooling_start_0")
        end = record.get("_infer_pooling_end_0")
        if start is not None or end is not None:
            if start is None or end is None:
                raise CapabilityError(
                    f"core60_mean requires paired explicit bounds for sequence view '{record.get('id')}'."
                )
            start_0 = int(start)
            end_0 = int(end)
            if start_0 < 0 or end_0 <= start_0 or end_0 > len(sequence) or (end_0 - start_0) != 60:
                raise CapabilityError(
                    "Sequence-view core60_mean received invalid explicit bounds: "
                    f"id={record.get('id')} start={start_0} end={end_0} length={len(sequence)}"
                )
            return start_0, end_0
        if len(sequence) != 60:
            raise CapabilityError(
                f"core60_mean requires an exact 60 bp sequence view. id={record.get('id')} length={len(sequence)}"
            )
        return 0, 60
    if operation != "anchor_mean":
        raise CapabilityError(f"Unsupported sequence-view pooling operation '{operation}'.")
    start = record.get("_infer_pooling_start_0")
    end = record.get("_infer_pooling_end_0")
    if start is None or end is None:
        raise CapabilityError(f"anchor_mean requires explicit bounds for sequence view '{record.get('id')}'.")
    start_0 = int(start)
    end_0 = int(end)
    if start_0 < 0 or end_0 <= start_0 or end_0 > len(sequence):
        raise CapabilityError(
            "Sequence-view anchor_mean received invalid emitted-orientation bounds: "
            f"id={record.get('id')} start={start_0} end={end_0} length={len(sequence)}"
        )
    return start_0, end_0


def load_sequence_view_input_records(
    *,
    bundle: SequenceFeatureBundleConfig,
) -> list[dict[str, object]]:
    result = load_sequence_view_input_records_with_status(bundle=bundle)
    if result.missing_products:
        first = result.missing_products[0]
        raise ValidationError(f"Sequence-view input selector resolved zero rows for dataset '{first.dataset}'.")
    return result.records


def load_sequence_view_input_records_with_status(
    *,
    bundle: SequenceFeatureBundleConfig,
) -> SequenceViewInputLoadResult:
    if not bundle.sequence_view_inputs:
        return SequenceViewInputLoadResult(records=[], missing_products=[])
    records: list[dict[str, object]] = []
    missing_products: list[SequenceViewMissingProduct] = []
    selected_cache: dict[tuple[str, str, str | None, str | None, str | None, str | None], list[dict[str, object]]] = {}
    sequence_cache: dict[tuple[str, str, str | None, str | None, str | None, str | None], dict[str, str]] = {}
    construct_cache: dict[
        tuple[str, str, str | None, str | None, str | None, str | None],
        dict[str, dict[str, object]],
    ] = {}
    for input_cfg in bundle.sequence_view_inputs:
        root = _resolve_usr_root(input_cfg.root)
        ds = Dataset(root, input_cfg.dataset)
        cache_key = (
            str(root),
            input_cfg.dataset,
            input_cfg.view_selector.product_kind,
            input_cfg.view_selector.view_name,
            input_cfg.view_selector.alias,
            input_cfg.view_selector.orientation,
        )
        selected = selected_cache.get(cache_key)
        if selected is None:
            selected = _select_sequence_view_rows(
                ds,
                product_kind=input_cfg.view_selector.product_kind,
                view_name=input_cfg.view_selector.view_name,
                alias=input_cfg.view_selector.alias,
                orientation=input_cfg.view_selector.orientation,
            )
            selected_cache[cache_key] = selected
        if not selected:
            missing_products.append(
                SequenceViewMissingProduct(
                    dataset=input_cfg.dataset,
                    root=str(root),
                    product_kind=input_cfg.view_selector.product_kind,
                    view_name=input_cfg.view_selector.view_name,
                    alias=input_cfg.view_selector.alias,
                    orientation=input_cfg.view_selector.orientation,
                    pooling_operation=input_cfg.pooling.operation,
                )
            )
            continue
        sequence_by_id = sequence_cache.get(cache_key)
        if sequence_by_id is None:
            sequence_by_id = _read_sequence_rows(ds, ids=sorted({str(row["sequence_id"]) for row in selected}))
            sequence_cache[cache_key] = sequence_by_id
        construct_by_id: dict[str, dict[str, object]] = {}
        if input_cfg.pooling.bounds_from == "construct_overlay":
            cached_construct_rows = construct_cache.get(cache_key)
            if cached_construct_rows is None:
                construct_by_id = _read_construct_rows(ds, ids=sorted({str(row["sequence_id"]) for row in selected}))
                construct_cache[cache_key] = construct_by_id
            else:
                construct_by_id = cached_construct_rows
        for view in selected:
            sequence_id = str(view["sequence_id"])
            sequence = sequence_by_id.get(sequence_id)
            if sequence is None:
                raise ValidationError(
                    f"Sequence-view input '{view['view_id']}' references missing sequence_id '{sequence_id}'."
                )
            construct_row = construct_by_id.get(sequence_id, {})
            pooling_start_0: int | None = None
            pooling_end_0: int | None = None
            if input_cfg.pooling.operation == "anchor_mean":
                if input_cfg.pooling.bounds_from == "construct_overlay":
                    pooling_start_0 = (
                        int(construct_row["construct__anchor_start"])
                        if construct_row.get("construct__anchor_start") not in {None, ""}
                        else None
                    )
                    pooling_end_0 = (
                        int(construct_row["construct__anchor_end"])
                        if construct_row.get("construct__anchor_end") not in {None, ""}
                        else None
                    )
                    if pooling_start_0 is None:
                        pooling_start_0 = view["anchor_start_0"]
                    if pooling_end_0 is None:
                        pooling_end_0 = view["anchor_end_0"]
                else:
                    pooling_start_0 = view["anchor_start_0"]
                    pooling_end_0 = view["anchor_end_0"]
            elif input_cfg.pooling.operation == "core60_mean" and input_cfg.pooling.start_0 is not None:
                pooling_start_0 = input_cfg.pooling.start_0
                pooling_end_0 = input_cfg.pooling.end_0
            record = {
                "id": view["view_id"],
                "sequence": sequence,
                "_infer_sequence_id": sequence_id,
                "_infer_view_id": view["view_id"],
                "_infer_view_name": view.get("view_name"),
                "_infer_product_kind": view["product_kind"],
                "_infer_context_kind": _context_kind_from_product_kind(
                    str(view["product_kind"]),
                    str(view["context_kind"]) if view.get("context_kind") not in {None, ""} else None,
                ),
                "_infer_orientation": view["orientation"],
                "_infer_parent_sequence_id": view.get("parent_sequence_id"),
                "_infer_derivation_id": view.get("derivation_id"),
                "_infer_source_dataset_id": input_cfg.dataset,
                "_infer_source_dataset_root": str(root),
                "_infer_anchor_start_0": view["anchor_start_0"],
                "_infer_anchor_end_0": view["anchor_end_0"],
                "_infer_pooling_operation": input_cfg.pooling.operation,
                "_infer_pooling_start_0": pooling_start_0,
                "_infer_pooling_end_0": pooling_end_0,
                "construct__anchor_start": construct_row.get("construct__anchor_start"),
                "construct__anchor_end": construct_row.get("construct__anchor_end"),
                "construct__anchor_id": construct_row.get("construct__anchor_id"),
                "construct__input_id": construct_row.get("construct__input_id"),
                "construct__anchor_orientation": construct_row.get("construct__anchor_orientation"),
                "construct__resolved_length": construct_row.get("construct__resolved_length"),
                "construct__spec_id": construct_row.get("construct__spec_id"),
                "construct__template_id": construct_row.get("construct__template_id"),
                "is_wildtype": construct_row.get("is_wildtype"),
            }
            _pooling_bounds_from_record(sequence=sequence, record=record)
            records.append(record)
    return SequenceViewInputLoadResult(records=records, missing_products=missing_products)


def resolve_sequence_view_contexts(*, records: list[dict[str, Any]]) -> list[SequenceContextRecord]:
    contexts: list[SequenceContextRecord] = []
    for row in records:
        sequence = str(row["sequence"])
        pooling_start_0, pooling_end_0 = _pooling_bounds_from_record(sequence=sequence, record=row)
        anchor_start = row.get("construct__anchor_start")
        anchor_end = row.get("construct__anchor_end")
        if anchor_start in {None, ""}:
            anchor_start = row.get("_infer_anchor_start_0")
        if anchor_end in {None, ""}:
            anchor_end = row.get("_infer_anchor_end_0")
        if anchor_start in {None, ""} and pooling_start_0 is not None:
            anchor_start = pooling_start_0
        if anchor_end in {None, ""} and pooling_end_0 is not None:
            anchor_end = pooling_end_0
        contexts.append(
            SequenceContextRecord(
                sequence_id=str(row["_infer_sequence_id"]),
                anchor_id=str(
                    row.get("construct__anchor_id") or row.get("_infer_parent_sequence_id") or row["_infer_sequence_id"]
                ),
                context_id=str(row["_infer_view_id"]),
                context_kind=str(row["_infer_context_kind"]),
                template_id=(
                    str(row.get("construct__template_id"))
                    if row.get("construct__template_id") not in {None, ""}
                    else None
                ),
                resolved_sequence=sequence,
                resolved_length=len(sequence),
                anchor_start=int(anchor_start) if anchor_start not in {None, ""} else 0,
                anchor_end=int(anchor_end) if anchor_end not in {None, ""} else len(sequence),
                anchor_orientation=str(
                    row.get("construct__anchor_orientation") or row.get("_infer_orientation") or "forward"
                ),
                construct_version=(
                    str(row.get("construct__spec_id")) if row.get("construct__spec_id") not in {None, ""} else None
                ),
                is_wildtype=_bool_or_none(row.get("is_wildtype")),
                view_id=str(row["_infer_view_id"]),
                view_name=str(row["_infer_view_name"]) if row.get("_infer_view_name") not in {None, ""} else None,
                product_kind=str(row["_infer_product_kind"]),
                orientation=str(row["_infer_orientation"]),
                parent_sequence_id=(
                    str(row["_infer_parent_sequence_id"])
                    if row.get("_infer_parent_sequence_id") not in {None, ""}
                    else None
                ),
                derivation_id=(
                    str(row["_infer_derivation_id"]) if row.get("_infer_derivation_id") not in {None, ""} else None
                ),
                source_dataset_id=str(row["_infer_source_dataset_id"]),
                source_dataset_root=str(row["_infer_source_dataset_root"]),
                pooling_operation=str(row["_infer_pooling_operation"]),
                pooling_start_0=pooling_start_0,
                pooling_end_0=pooling_end_0,
            )
        )
    return contexts
