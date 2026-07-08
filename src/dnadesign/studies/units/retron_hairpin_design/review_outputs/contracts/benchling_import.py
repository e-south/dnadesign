"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/review_outputs/contracts/benchling_import.py

Benchling import contract parsing for Retron hairpin review outputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

from ...compiler.exceptions import RetronMsdCompilerError
from .benchling_ids import parse_assigned_retron_ids, parse_source_precedent_ids

BENCHLING_GENBANK_DIRNAME = "benchling_genbank"
BENCHLING_ORIENTATION = "reverse_complement_only"
PAYLOAD_LABEL_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class BenchlingGenbankImportPlan:
    orientation: str
    expected_count: int
    filename_payload_label: str
    included_payload_trim_ids: tuple[str, ...]
    assigned_retron_ids: Mapping[str, str]
    source_precedent_ids: Mapping[str, str]
    expected_files: tuple[str, ...]

    @property
    def expected_variant_ids(self) -> tuple[str, ...]:
        return tuple(self.assigned_retron_ids)

    def assigned_construct_id(self, variant_id: str) -> str:
        try:
            return self.assigned_retron_ids[variant_id]
        except KeyError as exc:
            raise RetronMsdCompilerError(f"Retron Benchling import plan has no assigned id for {variant_id}") from exc

    def filename_for(self, variant_id: str) -> str:
        return f"{self.assigned_construct_id(variant_id)}-msd[{self.filename_payload_label}]-{variant_id}.gb"

    def source_precedent_id(self, variant_id: str) -> str:
        try:
            return self.source_precedent_ids[variant_id]
        except KeyError as exc:
            raise RetronMsdCompilerError(
                f"Retron Benchling import plan has no source precedent id for {variant_id}"
            ) from exc


def parse_benchling_genbank_import_plan(families: Mapping[str, object]) -> BenchlingGenbankImportPlan:
    raw_plan = _require_mapping(families.get("benchling_genbank_import"), "benchling_genbank_import")
    orientation = str(raw_plan.get("orientation") or "").strip()
    if orientation != BENCHLING_ORIENTATION:
        raise RetronMsdCompilerError(
            f"Retron Benchling GenBank import orientation must be {BENCHLING_ORIENTATION!r}, observed {orientation!r}"
        )
    assigned = parse_assigned_retron_ids(_require_mapping(raw_plan.get("assigned_retron_ids"), "assigned_retron_ids"))
    filename_payload_label = str(raw_plan.get("filename_payload_label") or "TetR").strip()
    if PAYLOAD_LABEL_RE.match(filename_payload_label) is None:
        raise RetronMsdCompilerError(f"Retron Benchling filename_payload_label is invalid: {filename_payload_label}")
    expected_files = _require_string_list(raw_plan.get("expected_files"), "expected_files")
    expected_from_ids = tuple(
        f"{BENCHLING_GENBANK_DIRNAME}/{construct_id}-msd[{filename_payload_label}]-{variant_id}.gb"
        for variant_id, construct_id in assigned.items()
    )
    if expected_files != expected_from_ids:
        raise RetronMsdCompilerError(
            "Retron Benchling GenBank expected_files must match assigned_retron_ids in order: "
            f"{list(expected_files)} != {list(expected_from_ids)}"
        )
    expected_count = int(raw_plan.get("expected_count") or 0)
    if expected_count != len(assigned):
        raise RetronMsdCompilerError(
            f"Retron Benchling GenBank expected_count {expected_count} does not match assigned id count {len(assigned)}"
        )
    precedents = parse_source_precedent_ids(
        _require_mapping(raw_plan.get("source_precedent_ids"), "source_precedent_ids")
    )
    if tuple(precedents) != tuple(assigned):
        raise RetronMsdCompilerError(
            "Retron Benchling GenBank source_precedent_ids must match assigned_retron_ids in order: "
            f"{list(precedents)} != {list(assigned)}"
        )
    included = _require_string_list(raw_plan.get("included_payload_trim_ids"), "included_payload_trim_ids")
    if not included:
        raise RetronMsdCompilerError("Retron Benchling GenBank import must include at least one payload trim id")
    return BenchlingGenbankImportPlan(
        orientation=orientation,
        expected_count=expected_count,
        filename_payload_label=filename_payload_label,
        included_payload_trim_ids=included,
        assigned_retron_ids=assigned,
        source_precedent_ids=precedents,
        expected_files=expected_files,
    )


def _require_mapping(raw: object, label: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping):
        raise RetronMsdCompilerError(f"Retron Benchling GenBank import expected mapping for {label}")
    return raw


def _require_string_list(raw: object, label: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not all(isinstance(item, str) and item.strip() for item in raw):
        raise RetronMsdCompilerError(f"Retron Benchling GenBank import expected non-empty string list for {label}")
    return tuple(item.strip() for item in raw)


__all__ = ["BENCHLING_GENBANK_DIRNAME", "BenchlingGenbankImportPlan", "parse_benchling_genbank_import_plan"]
