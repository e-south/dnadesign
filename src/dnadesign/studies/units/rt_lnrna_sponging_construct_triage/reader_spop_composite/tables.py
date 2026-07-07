"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reader_spop_composite/tables.py

Parquet table writers for the RT-lnRNA Reader SPOP composite.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

CONDITION_MATRIX_TABLE = "reader_spop_condition_matrix.parquet"
CONDITION_COLUMNS_TABLE = "reader_spop_condition_columns.parquet"


class _TableRow(Protocol):
    def to_dict(self) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class ReaderSpopConditionMatrixTables:
    output_dir: str
    condition_matrix_path: str
    condition_columns_path: str
    row_count: int
    condition_count: int
    missing_cell_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def write_condition_matrix_tables(
    *,
    rows: Sequence[_TableRow],
    condition_columns: Sequence[_TableRow],
    output_dir: Path,
    missing_cell_count: int,
) -> ReaderSpopConditionMatrixTables:
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = resolved_output_dir / CONDITION_MATRIX_TABLE
    columns_path = resolved_output_dir / CONDITION_COLUMNS_TABLE
    pq.write_table(_condition_row_table(rows), matrix_path)
    pq.write_table(_condition_column_table(condition_columns), columns_path)
    return ReaderSpopConditionMatrixTables(
        output_dir=resolved_output_dir.as_posix(),
        condition_matrix_path=matrix_path.as_posix(),
        condition_columns_path=columns_path.as_posix(),
        row_count=len(rows),
        condition_count=len(condition_columns),
        missing_cell_count=missing_cell_count,
    )


def _condition_row_table(rows: Sequence[_TableRow]) -> pa.Table:
    return pa.Table.from_pylist([row.to_dict() for row in rows], schema=_condition_row_schema())


def _condition_column_table(rows: Sequence[_TableRow]) -> pa.Table:
    return pa.Table.from_pylist([row.to_dict() for row in rows], schema=_condition_column_schema())


def _condition_row_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("observation_id", pa.string()),
            pa.field("assay_subject_key", pa.string()),
            pa.field("reader_design_id", pa.string()),
            pa.field("reader_experiment_id", pa.string()),
            pa.field("condition_key", pa.string()),
            pa.field("condition_role", pa.string()),
            pa.field("atc_nM", pa.float64()),
            pa.field("iptg_uM", pa.float64()),
            pa.field("normalized_derepression", pa.float64()),
            pa.field("rfp_over_od600", pa.float64()),
            pa.field("viability_relative_to_baseline", pa.float64()),
            pa.field("replicate_count", pa.int64()),
            pa.field("construct_subject_id", pa.string()),
            pa.field("construct_subject_bridge_status", pa.string()),
            pa.field("qc_flags", pa.list_(pa.string())),
            pa.field("value_basis", pa.string()),
        ]
    )


def _condition_column_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("condition_key", pa.string()),
            pa.field("condition_role", pa.string()),
            pa.field("atc_nM", pa.float64()),
            pa.field("iptg_uM", pa.float64()),
        ]
    )
