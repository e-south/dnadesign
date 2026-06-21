"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/review_outputs/sequence_index.py

Materialized sequence-index validation for Retron review outputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..artifact_contracts.layout import MANIFEST_DIRNAME, MANIFEST_INDEXES_DIRNAME, SEQUENCE_INDEX_FILENAME
from ..compiler.exceptions import RetronMsdCompilerError
from .plan import TetoReviewPlan


@dataclass(frozen=True)
class SequenceReviewFrame:
    order: int
    construct_id: str
    msd_design_id: str
    payload_trim_id: str
    scaffold_context: str
    variant_role: str
    rt_mode: str
    composition_overview_png: Path
    row: Mapping[str, str]

    @property
    def label(self) -> str:
        return (
            f"{self.construct_id} | {self.msd_design_id} | {self.payload_trim_id} | "
            f"{self.scaffold_context} | {self.variant_role}"
        )


REQUIRED_ROW_ARTIFACT_FIELDS = (
    "genbank",
    "reverse_complement_genbank",
    "forward_fasta",
    "reverse_complement_fasta",
    "features_csv",
    "composition_overview_svg",
    "composition_overview_png",
    "secondary_structure_native_png",
)


def load_validated_sequence_frames(materialized_root: Path, *, plan: TetoReviewPlan) -> tuple[SequenceReviewFrame, ...]:
    root = materialized_root.expanduser().resolve()
    index_path = root / MANIFEST_DIRNAME / MANIFEST_INDEXES_DIRNAME / SEQUENCE_INDEX_FILENAME
    if not index_path.is_file():
        raise RetronMsdCompilerError(f"Retron review sequence index not found: {index_path}")
    rows = _read_tsv(index_path)
    if len(rows) != plan.expected_variant_count:
        raise RetronMsdCompilerError(
            f"Expected {plan.expected_variant_count} materialized sequence rows for "
            f"{plan.deliverable_plan_id}, found {len(rows)} in {index_path}"
        )

    expected_trim_ids = {panel.payload_trim_id for panel in plan.pwm_panels}
    observed_trim_ids = {row.get("payload_trim_id", "") for row in rows}
    if observed_trim_ids != expected_trim_ids:
        raise RetronMsdCompilerError(
            "Retron review sequence_index.tsv payload_trim_id set does not match the PWM triptych panels: "
            f"{sorted(observed_trim_ids)} != {sorted(expected_trim_ids)}"
        )
    frames = []
    for order, row in enumerate(rows, start=1):
        _validate_row(row, root=root, index_path=index_path, order=order)
        frames.append(
            SequenceReviewFrame(
                order=order,
                construct_id=_require_cell(row, "construct_id", order=order),
                msd_design_id=_require_cell(row, "msd_design_id", order=order),
                payload_trim_id=_require_cell(row, "payload_trim_id", order=order),
                scaffold_context=_require_cell(row, "scaffold_context", order=order),
                variant_role=_require_cell(row, "variant_role", order=order),
                rt_mode=_require_cell(row, "rt_mode", order=order),
                composition_overview_png=root / _require_cell(row, "composition_overview_png", order=order),
                row=row,
            )
        )
    return tuple(frames)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [dict(row) for row in reader]


def _validate_row(row: Mapping[str, str], *, root: Path, index_path: Path, order: int) -> None:
    if row.get("folding_status") != "ok":
        raise RetronMsdCompilerError(
            f"Retron review output requires folding_status == ok for row {order} in {index_path}; "
            f"observed {row.get('folding_status')!r}"
        )
    for field in REQUIRED_ROW_ARTIFACT_FIELDS:
        rel_path = _require_cell(row, field, order=order)
        path = root / rel_path
        if not path.is_file():
            raise RetronMsdCompilerError(f"Missing materialized review artifact for row {order} field {field}: {path}")


def _require_cell(row: Mapping[str, str], field: str, *, order: int) -> str:
    value = str(row.get(field) or "").strip()
    if not value:
        raise RetronMsdCompilerError(f"Retron review sequence_index.tsv row {order} is missing {field}")
    return value


__all__ = ["REQUIRED_ROW_ARTIFACT_FIELDS", "SequenceReviewFrame", "load_validated_sequence_frames"]
