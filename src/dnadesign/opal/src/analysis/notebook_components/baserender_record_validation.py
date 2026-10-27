"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/baserender_record_validation.py

Public-adapter validation for BaseRender notebook record discovery.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Iterable, Mapping

from .baserender_record_sources import (
    contract_valid_filters,
    id_column,
    join_metadata_rows,
    metadata_records_path,
    normalise_record_ids,
    record_source_columns,
    require_unique_record_ids,
)


def public_adapter_valid_record_ids(
    records_path: str | Path,
    contract: Mapping[str, Any],
    *,
    record_ids: Iterable[Any],
) -> list[str]:
    """Keep record ids whose rows satisfy the contract's declared public adapter."""

    candidate_ids = normalise_record_ids(record_ids)
    if not candidate_ids:
        return []

    import polars as pl

    identifier = id_column(contract)
    source_columns = record_source_columns(contract)
    if identifier not in source_columns:
        source_columns.append(identifier)
    scan = pl.scan_parquet(str(records_path)).select(source_columns)
    scan = scan.filter(pl.col(identifier).cast(pl.Utf8).is_in(candidate_ids))
    require_unique_record_ids(pl, scan, id_column_name=identifier)
    schema = scan.collect_schema()
    for expr in contract_valid_filters(pl, contract, schema):
        scan = scan.filter(expr)
    metadata_path = metadata_records_path(contract)
    if metadata_path is not None:
        scan = join_metadata_rows(pl, scan, metadata_path, id_column_name=identifier, contract=contract)
    require_unique_record_ids(pl, scan, id_column_name=identifier)
    rows = scan.collect().to_dicts()
    rows_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        record_id = str(row.get(identifier) or "").strip()
        if record_id:
            rows_by_id.setdefault(record_id, []).append(row)

    baserender = import_module("dnadesign.baserender")
    adapter_kind = str(contract.get("adapter_kind") or "").strip()
    adapter_columns = contract.get("adapter_columns")
    adapter_policies = contract.get("adapter_policies")
    valid_ids: list[str] = []
    for record_id in candidate_ids:
        for row in rows_by_id.get(record_id, ()):
            try:
                baserender.adapt_record(
                    row,
                    adapter_kind=adapter_kind,
                    adapter_columns=dict(adapter_columns) if isinstance(adapter_columns, Mapping) else None,
                    adapter_policies=dict(adapter_policies) if isinstance(adapter_policies, Mapping) else None,
                )
            except baserender.SchemaError:
                continue
            valid_ids.append(record_id)
            break
    return valid_ids


__all__ = ["public_adapter_valid_record_ids"]
