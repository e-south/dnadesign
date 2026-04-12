"""
Alignment artifact builders for latentdna.
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


def _key_columns(alignment: AlignmentConfig, *, record_key: str, subject_key: str) -> list[str]:
    if isinstance(alignment.on, list):
        return alignment.on
    if alignment.on == "record_key":
        return [record_key]
    if alignment.on == "subject_key":
        return [subject_key]
    raise ContractViolationError(f"unsupported alignment key basis: {alignment.on!r}")


def _load_alignment_input(context: WorkspaceContext, ref_id: str, alignment: AlignmentConfig) -> AlignmentInput:
    if ref_id in context.config.sources:
        source = context.require_source(ref_id)
        resolved = resolve_source(ref_id, source, workspace_dir=context.workspace_dir)
        key_columns = _key_columns(alignment, record_key=source.record_key, subject_key=source.subject_key)
        rows = read_records_table(resolved, columns=key_columns)
        if resolved.records_path is None:
            raise ContractViolationError(f"source {ref_id} does not expose a records table for alignment")
        return AlignmentInput(ref_id=ref_id, rows=rows, input_path=resolved.records_path, key_columns=key_columns)

    view = context.require_source_view(ref_id)
    source = context.require_source(view.source)
    rows_path = context.output_root / "views" / ref_id / "rows.parquet"
    if not rows_path.exists():
        raise MissingArtifactError(f"alignment input view is not materialized: {ref_id}")
    key_columns = _key_columns(alignment, record_key=source.record_key, subject_key=source.subject_key)
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
) -> tuple[Path, int, int, int, Path, Path, list[str]]:
    alignment = context.require_alignment(alignment_id)
    left = _load_alignment_input(context, alignment.left, alignment)
    right = _load_alignment_input(context, alignment.right, alignment)
    if left.key_columns != right.key_columns:
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
    )
