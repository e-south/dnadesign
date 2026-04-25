"""
Sample building helpers for latentdna.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from ..contracts.errors import ContractViolationError
from ..io.parquet_io import read_table, write_table
from ..workspaces.loader import WorkspaceContext


def _allocate_stratified_counts(groups: dict[str, list[int]], total: int) -> dict[str, int]:
    total_rows = sum(len(indices) for indices in groups.values())
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    assigned = 0
    for key, indices in groups.items():
        raw = (len(indices) / total_rows) * total
        count = min(len(indices), int(raw))
        quotas[key] = count
        assigned += count
        remainders.append((raw - count, key))
    for _, key in sorted(remainders, reverse=True):
        if assigned >= total:
            break
        if quotas[key] >= len(groups[key]):
            continue
        quotas[key] += 1
        assigned += 1
    return quotas


def _sample_rows_path(context: WorkspaceContext, sample_id: str) -> Path:
    path = context.output_root / "samples" / sample_id / "rows.parquet"
    if not path.exists():
        raise ContractViolationError(f"sample rows are missing for sample source {sample_id}: {path}")
    return path


def _row_identity(row: dict[str, Any], columns: list[str]) -> tuple[Any, ...]:
    return tuple(row[column] for column in columns)


def _combine_sample_sets(
    context: WorkspaceContext,
    *,
    strategy: str,
    input_sample_ids: list[str],
) -> pa.Table:
    if len(input_sample_ids) < 2:
        raise ContractViolationError(f"{strategy} sampling requires at least two --input-sample values")

    sample_tables = [read_table(_sample_rows_path(context, sample_id)) for sample_id in input_sample_ids]
    column_names = list(sample_tables[0].column_names)
    for sample_id, table in zip(input_sample_ids[1:], sample_tables[1:], strict=True):
        if list(table.column_names) != column_names:
            raise ContractViolationError(
                f"{strategy} sampling requires matching row-ledger schemas across inputs; {sample_id!r} differs"
            )

    row_groups = [table.to_pylist() for table in sample_tables]
    if strategy == "union":
        seen: set[tuple[Any, ...]] = set()
        output_rows: list[dict[str, Any]] = []
        for rows in row_groups:
            for row in rows:
                identity = _row_identity(row, column_names)
                if identity in seen:
                    continue
                seen.add(identity)
                output_rows.append(row)
        return pa.Table.from_pylist(output_rows, schema=sample_tables[0].schema)

    shared = {_row_identity(row, column_names) for row in row_groups[0]}
    for rows in row_groups[1:]:
        shared &= {_row_identity(row, column_names) for row in rows}
    output_rows = [row for row in row_groups[0] if _row_identity(row, column_names) in shared]
    return pa.Table.from_pylist(output_rows, schema=sample_tables[0].schema)


def build_sample_artifact(
    context: WorkspaceContext,
    *,
    sample_id: str,
    view_id: str | None,
    strategy: str,
    n: int | None,
    group_column: str | None,
    seed: int,
    explicit_ids: list[str] | None = None,
    input_sample_ids: list[str] | None = None,
    reference_set_id: str | None = None,
) -> tuple[Path, int]:
    if strategy in {"union", "intersection"}:
        sample_table = _combine_sample_sets(
            context,
            strategy=strategy,
            input_sample_ids=input_sample_ids or [],
        )
    else:
        if view_id is None:
            raise ContractViolationError(f"{strategy} sampling requires --view")
        view_dir = context.output_root / "views" / view_id
        rows_path = view_dir / "rows.parquet"
        if not rows_path.exists():
            raise ContractViolationError(f"view rows are missing for sample source {view_id}: {rows_path}")
        rows = read_table(rows_path)
        if rows.num_rows == 0:
            raise ContractViolationError(f"view {view_id} has no rows to sample")

        rng = np.random.default_rng(seed)
        selected_indices: list[int]
        row_count = rows.num_rows
        if strategy == "all":
            selected_indices = list(range(row_count))
        elif strategy == "random":
            if n is None:
                raise ContractViolationError("random sampling requires --n")
            selected_indices = sorted(rng.choice(row_count, size=min(n, row_count), replace=False).tolist())
        elif strategy == "stratified":
            if n is None:
                raise ContractViolationError("stratified sampling requires --n")
            if not group_column:
                raise ContractViolationError("stratified sampling requires --group-column")
            if group_column not in rows.column_names:
                raise ContractViolationError(f"stratified sampling column is missing from row ledger: {group_column!r}")
            groups: dict[str, list[int]] = defaultdict(list)
            values = rows[group_column].combine_chunks().to_pylist()
            for index, value in enumerate(values):
                groups[str(value)].append(index)
            quotas = _allocate_stratified_counts(groups, min(n, row_count))
            selected_indices = []
            for key in sorted(groups):
                candidates = np.asarray(groups[key], dtype=int)
                order = rng.permutation(len(candidates))
                chosen = sorted(candidates[order][: quotas[key]].tolist())
                selected_indices.extend(chosen)
            selected_indices = sorted(selected_indices)
        elif strategy == "explicit_ids":
            requested_ids = [str(value) for value in explicit_ids or []]
            if not requested_ids:
                raise ContractViolationError("explicit_ids sampling requires at least one explicit id")
            key_column = rows.column_names[0]
            values = [str(value) for value in rows[key_column].combine_chunks().to_pylist()]
            requested = set(requested_ids)
            selected_indices = [index for index, value in enumerate(values) if value in requested]
            found_ids = {values[index] for index in selected_indices}
            missing_ids = sorted(requested.difference(found_ids))
            if missing_ids:
                missing = ", ".join(missing_ids)
                raise ContractViolationError(f"explicit_ids sampling could not find ids in {key_column!r}: {missing}")
        else:
            raise ContractViolationError(f"unsupported sampling strategy: {strategy}")

        if reference_set_id is not None:
            if reference_set_id not in context.config.reference_sets:
                raise ContractViolationError(f"unknown reference_set for sampling: {reference_set_id}")
            reference_set = context.config.reference_sets[reference_set_id]
            match_column = reference_set.match_column
            if match_column not in rows.column_names:
                raise ContractViolationError(f"reference_set match column is missing from row ledger: {match_column!r}")
            value_to_index: dict[str, int] = {}
            for index, value in enumerate(rows[match_column].combine_chunks().to_pylist()):
                value_to_index.setdefault(str(value), index)
            missing_ids = [str(value) for value in reference_set.ids if str(value) not in value_to_index]
            if missing_ids:
                missing = ", ".join(missing_ids)
                raise ContractViolationError(
                    f"reference_set sampling could not find ids in {match_column!r}: {missing}"
                )
            selected_indices = sorted(
                {
                    *selected_indices,
                    *(value_to_index[str(value)] for value in reference_set.ids),
                }
            )

        if selected_indices:
            sample_table = rows.take(pa.array(selected_indices, type=pa.int64()))
        else:
            sample_table = rows.slice(0, 0)
    artifact_dir = context.output_root / "samples" / sample_id
    write_table(sample_table, artifact_dir / "rows.parquet")
    return artifact_dir, sample_table.num_rows
