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

from ..reader_spop_plan import ReaderSpopObservation, ReaderSpopPlan
from .conditions import (
    BASELINE_CONDITION_KEY,
    BASELINE_ROLE,
    IPTG_DOSE_ROLE,
    POSITIVE_CONTROL_ROLE,
    condition_key_for_iptg_dose,
    condition_key_for_positive_control,
    condition_sort_key,
)
from .identifiers import variant_sort_key
from .tables import ReaderSpopConditionMatrixTables, write_condition_matrix_tables


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
                condition_sort_key(
                    condition_role=row.condition_role,
                    atc_nM=row.atc_nM,
                    iptg_uM=row.iptg_uM,
                    condition_key=row.condition_key,
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

    return write_condition_matrix_tables(
        rows=matrix.rows,
        condition_columns=matrix.condition_columns,
        output_dir=output_dir,
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
            condition_key=BASELINE_CONDITION_KEY,
            condition_role=BASELINE_ROLE,
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
            condition_key=condition_key_for_positive_control(observation.positive_control_atc_nM),
            condition_role=POSITIVE_CONTROL_ROLE,
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
                condition_key=condition_key_for_iptg_dose(dose),
                condition_role=IPTG_DOSE_ROLE,
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
    return tuple(
        sorted(
            by_key.values(),
            key=lambda column: condition_sort_key(
                condition_role=column.condition_role,
                atc_nM=column.atc_nM,
                iptg_uM=column.iptg_uM,
                condition_key=column.condition_key,
            ),
        )
    )
