"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reader_spop_composite/condition_matrix.py

Study-owned condition-long Reader SPOP matrix for RT-lnRNA sponging triage.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from ..reader_spop_plan import ReaderSpopObservation, ReaderSpopPlan
from .identifiers import variant_sort_key

CONDITION_MATRIX_TABLE = "reader_spop_condition_matrix.parquet"
CONDITION_COLUMNS_TABLE = "reader_spop_condition_columns.parquet"


@dataclass(frozen=True, slots=True)
class ReaderSpopConditionColumn:
    condition_key: str
    condition_role: str
    atc_nM: float
    iptg_uM: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReaderSpopConditionRow:
    observation_id: str
    assay_subject_key: str
    reader_design_id: str
    reader_experiment_id: str
    condition_key: str
    condition_role: str
    atc_nM: float
    iptg_uM: float
    normalized_derepression: float
    rfp_over_od600: float
    viability_relative_to_baseline: float | None
    replicate_count: int
    construct_subject_id: str | None
    construct_subject_bridge_status: str
    qc_flags: tuple[str, ...]
    value_basis: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["qc_flags"] = list(self.qc_flags)
        return payload


@dataclass(frozen=True, slots=True)
class ReaderSpopConditionMatrix:
    rows: tuple[ReaderSpopConditionRow, ...]
    condition_columns: tuple[ReaderSpopConditionColumn, ...]
    missing_cell_count: int
    source_reader_experiment_ids: tuple[str, ...]

    @property
    def variant_count(self) -> int:
        return len({row.assay_subject_key for row in self.rows})


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


def build_reader_spop_condition_matrix(plan: ReaderSpopPlan) -> ReaderSpopConditionMatrix:
    """Expand Reader SPOP observations into sparse condition-long evidence rows."""

    rows: list[ReaderSpopConditionRow] = []
    for observation in plan.observations:
        rows.extend(_condition_rows_for_observation(observation))
    ordered_rows = tuple(
        sorted(
            rows,
            key=lambda row: (
                variant_sort_key(row.assay_subject_key),
                row.reader_experiment_id,
                _condition_sort_key(
                    ReaderSpopConditionColumn(
                        condition_key=row.condition_key,
                        condition_role=row.condition_role,
                        atc_nM=row.atc_nM,
                        iptg_uM=row.iptg_uM,
                    )
                ),
            ),
        )
    )
    condition_columns = _condition_columns(ordered_rows)
    observed_cells = {(row.assay_subject_key, row.condition_key) for row in ordered_rows}
    variants = {row.assay_subject_key for row in ordered_rows}
    missing_cell_count = len(variants) * len(condition_columns) - len(observed_cells)
    return ReaderSpopConditionMatrix(
        rows=ordered_rows,
        condition_columns=condition_columns,
        missing_cell_count=missing_cell_count,
        source_reader_experiment_ids=tuple(sorted({row.reader_experiment_id for row in ordered_rows})),
    )


def write_reader_spop_condition_matrix(
    matrix: ReaderSpopConditionMatrix,
    *,
    output_dir: Path,
) -> ReaderSpopConditionMatrixTables:
    """Write the condition matrix and its explicit condition-column catalog."""

    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = resolved_output_dir / CONDITION_MATRIX_TABLE
    columns_path = resolved_output_dir / CONDITION_COLUMNS_TABLE
    pq.write_table(_condition_row_table(matrix.rows), matrix_path)
    pq.write_table(_condition_column_table(matrix.condition_columns), columns_path)
    return ReaderSpopConditionMatrixTables(
        output_dir=resolved_output_dir.as_posix(),
        condition_matrix_path=matrix_path.as_posix(),
        condition_columns_path=columns_path.as_posix(),
        row_count=len(matrix.rows),
        condition_count=len(matrix.condition_columns),
        missing_cell_count=matrix.missing_cell_count,
    )


def _condition_rows_for_observation(observation: ReaderSpopObservation) -> list[ReaderSpopConditionRow]:
    baseline = float(observation.rfp_over_od600_baseline)
    positive = float(observation.rfp_over_od600_positive)
    spread = positive - baseline
    rows = [
        ReaderSpopConditionRow(
            observation_id=observation.observation_id,
            assay_subject_key=observation.assay_subject_key,
            reader_design_id=observation.reader_design_id,
            reader_experiment_id=observation.reader_experiment_id,
            condition_key="0 nm aTc; 0 uM IPTG",
            condition_role="baseline",
            atc_nM=0.0,
            iptg_uM=0.0,
            normalized_derepression=0.0,
            rfp_over_od600=baseline,
            viability_relative_to_baseline=None,
            replicate_count=int(observation.replicate_count),
            construct_subject_id=observation.construct_subject_id,
            construct_subject_bridge_status=observation.construct_subject_bridge_status,
            qc_flags=observation.qc_flags,
            value_basis="reader_spop_direct_baseline",
        ),
        ReaderSpopConditionRow(
            observation_id=observation.observation_id,
            assay_subject_key=observation.assay_subject_key,
            reader_design_id=observation.reader_design_id,
            reader_experiment_id=observation.reader_experiment_id,
            condition_key=f"{observation.positive_control_atc_nM:g} nm aTc; 0 uM IPTG",
            condition_role="positive_control",
            atc_nM=float(observation.positive_control_atc_nM),
            iptg_uM=0.0,
            normalized_derepression=1.0,
            rfp_over_od600=positive,
            viability_relative_to_baseline=None,
            replicate_count=int(observation.replicate_count),
            construct_subject_id=observation.construct_subject_id,
            construct_subject_bridge_status=observation.construct_subject_bridge_status,
            qc_flags=observation.qc_flags,
            value_basis="reader_spop_direct_positive_control",
        ),
    ]
    for dose, y_value, viability in zip(
        observation.iptg_doses_uM,
        observation.y_derepression_by_dose,
        observation.viability_by_dose,
        strict=True,
    ):
        normalized = float(y_value)
        rows.append(
            ReaderSpopConditionRow(
                observation_id=observation.observation_id,
                assay_subject_key=observation.assay_subject_key,
                reader_design_id=observation.reader_design_id,
                reader_experiment_id=observation.reader_experiment_id,
                condition_key=f"0 nm aTc; {float(dose):g} uM IPTG",
                condition_role="iptg_dose",
                atc_nM=0.0,
                iptg_uM=float(dose),
                normalized_derepression=normalized,
                rfp_over_od600=baseline + normalized * spread,
                viability_relative_to_baseline=float(viability),
                replicate_count=int(observation.replicate_count),
                construct_subject_id=observation.construct_subject_id,
                construct_subject_bridge_status=observation.construct_subject_bridge_status,
                qc_flags=observation.qc_flags,
                value_basis="reader_spop_reconstructed_from_normalized_endpoint",
            )
        )
    return rows


def _condition_columns(rows: Sequence[ReaderSpopConditionRow]) -> tuple[ReaderSpopConditionColumn, ...]:
    by_key: dict[str, ReaderSpopConditionColumn] = {}
    for row in rows:
        by_key.setdefault(
            row.condition_key,
            ReaderSpopConditionColumn(
                condition_key=row.condition_key,
                condition_role=row.condition_role,
                atc_nM=float(row.atc_nM),
                iptg_uM=float(row.iptg_uM),
            ),
        )
    return tuple(sorted(by_key.values(), key=_condition_sort_key))


def _condition_sort_key(column: ReaderSpopConditionColumn) -> tuple[int, float, float, str]:
    role_order = {"baseline": 0, "positive_control": 1, "iptg_dose": 2}
    return (
        role_order.get(column.condition_role, 99),
        float(column.atc_nM) if column.condition_role == "positive_control" else 0.0,
        float(column.iptg_uM),
        column.condition_key,
    )


def _condition_row_table(rows: Sequence[ReaderSpopConditionRow]) -> pa.Table:
    return pa.Table.from_pylist([row.to_dict() for row in rows], schema=_condition_row_schema())


def _condition_column_table(rows: Sequence[ReaderSpopConditionColumn]) -> pa.Table:
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
