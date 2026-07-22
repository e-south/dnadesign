"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/alignments/builder.py

Alignment artifact builders for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa

from ..contracts.errors import AlignmentError, ContractViolationError, MissingArtifactError
from ..contracts.workspace import AlignmentConfig
from ..io.parquet_io import read_table, write_table
from ..sources.resolver import read_records_table, resolve_source
from ..workspaces.loader import WorkspaceContext


@dataclass(frozen=True, slots=True)
class AlignmentInput:
    ref_id: str
    rows: pa.Table
    input_path: Path
    key_columns: list[str]


def _shared_key_columns(alignment: AlignmentConfig, *, record_key: str, subject_key: str) -> list[str]:
    if alignment.on is None:
        raise ContractViolationError("alignment key basis is undefined")
    if isinstance(alignment.on, list):
        return alignment.on
    if alignment.on == "record_key":
        return [record_key]
    if alignment.on == "subject_key":
        return [subject_key]
    raise ContractViolationError(f"unsupported alignment key basis: {alignment.on!r}")


def _key_columns_for_side(
    alignment: AlignmentConfig,
    *,
    side: str,
    record_key: str,
    subject_key: str,
) -> list[str]:
    if alignment.left_on is not None and alignment.right_on is not None:
        return alignment.left_on if side == "left" else alignment.right_on
    return _shared_key_columns(alignment, record_key=record_key, subject_key=subject_key)


def _load_alignment_input(
    context: WorkspaceContext,
    ref_id: str,
    alignment: AlignmentConfig,
    *,
    side: str,
) -> AlignmentInput:
    if ref_id in context.config.sources:
        source = context.require_source(ref_id)
        resolved = resolve_source(ref_id, source, workspace_dir=context.workspace_dir)
        key_columns = _key_columns_for_side(
            alignment,
            side=side,
            record_key=source.record_key,
            subject_key=source.subject_key,
        )
        rows = read_records_table(resolved, columns=key_columns)
        if resolved.records_path is None:
            raise ContractViolationError(f"source {ref_id} does not expose a records table for alignment")
        return AlignmentInput(ref_id=ref_id, rows=rows, input_path=resolved.records_path, key_columns=key_columns)

    view = context.require_view(ref_id)
    rows_path = context.output_root / "views" / ref_id / "rows.parquet"
    if not rows_path.exists():
        raise MissingArtifactError(f"alignment input view is not materialized: {ref_id}")
    view_source_id = getattr(view, "source", None)
    if view_source_id is not None:
        source = context.require_source(str(view_source_id))
        record_key = source.record_key
        subject_key = source.subject_key
    else:
        manifest_path = rows_path.parent / "manifest.json"
        if not manifest_path.exists():
            raise MissingArtifactError(f"alignment input derived view is missing manifest: {ref_id}")
        manifest = context.read_manifest(manifest_path)
        params = dict(manifest.get("params") or {})
        record_key = str(params.get("record_key") or "").strip()
        subject_key = str(params.get("subject_key") or record_key).strip()
        if not record_key:
            raise ContractViolationError(f"derived view {ref_id} manifest does not declare a record_key")
    key_columns = _key_columns_for_side(
        alignment,
        side=side,
        record_key=record_key,
        subject_key=subject_key,
    )
    rows = read_table(rows_path, columns=key_columns)
    return AlignmentInput(ref_id=ref_id, rows=rows, input_path=rows_path, key_columns=key_columns)


def _group_indices(table: pa.Table, *, key_columns: list[str]) -> dict[tuple[Any, ...], list[int]]:
    grouped: dict[tuple[Any, ...], list[int]] = {}
    for index, row in enumerate(table.to_pylist()):
        key = tuple(row[name] for name in key_columns)
        grouped.setdefault(key, []).append(index)
    return grouped


def _represent_key_row(key_columns: list[str], key: tuple[Any, ...]) -> dict[str, Any]:
    return {name: value for name, value in zip(key_columns, key, strict=True)}


def _require_supported_multiplicity(groups: dict[tuple[Any, ...], list[int]], *, mode: str, label: str) -> None:
    if mode != "error":
        return
    duplicates = [(key, len(indices)) for key, indices in groups.items() if len(indices) > 1]
    if duplicates:
        key, count = duplicates[0]
        raise AlignmentError(f"{label} is non-unique on the alignment keys: {key!r} matched {count} rows")


def build_alignment_artifact(
    context: WorkspaceContext,
    *,
    alignment_id: str,
) -> tuple[Path, int, int, int, Path, Path, list[str], list[str]]:
    alignment = context.require_alignment(alignment_id)
    left = _load_alignment_input(context, alignment.left, alignment, side="left")
    right = _load_alignment_input(context, alignment.right, alignment, side="right")
    if len(left.key_columns) != len(right.key_columns):
        raise AlignmentError(
            f"alignment {alignment_id} resolved mismatched key columns: {left.key_columns!r} vs {right.key_columns!r}"
        )

    left_groups = _group_indices(left.rows, key_columns=left.key_columns)
    right_groups = _group_indices(right.rows, key_columns=right.key_columns)
    _require_supported_multiplicity(
        left_groups,
        mode=alignment.left_aggregation,
        label=f"alignment {alignment_id} left",
    )
    _require_supported_multiplicity(
        right_groups,
        mode=alignment.right_aggregation,
        label=f"alignment {alignment_id} right",
    )
    common_keys = sorted(set(left_groups).intersection(right_groups))
    if not common_keys:
        raise AlignmentError(f"alignment {alignment_id} produced empty intersection support")

    mapping_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    for key in common_keys:
        left_indices = left_groups[key]
        right_indices = right_groups[key]
        ledger_row = {
            **_represent_key_row(left.key_columns, key),
            "left_count": len(left_indices),
            "right_count": len(right_indices),
        }
        ledger_rows.append(ledger_row)
        mapping_rows.append(
            {
                **ledger_row,
                "left_indices": left_indices,
                "right_indices": right_indices,
            }
        )

    artifact_dir = context.output_root / "alignments" / alignment_id
    write_table(pa.Table.from_pylist(ledger_rows), artifact_dir / "rows.parquet")
    write_table(pa.Table.from_pylist(mapping_rows), artifact_dir / "mapping.parquet")

    left_unmatched = sum(len(indices) for key, indices in left_groups.items() if key not in right_groups)
    right_unmatched = sum(len(indices) for key, indices in right_groups.items() if key not in left_groups)
    return (
        artifact_dir,
        len(common_keys),
        left_unmatched,
        right_unmatched,
        left.input_path,
        right.input_path,
        left.key_columns,
        right.key_columns,
    )
