"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/review_outputs/contracts/benchling_import.py

Benchling import contract parsing for Retron hairpin review outputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ...compiler.exceptions import RetronMsdCompilerError
from .benchling_ids import parse_assigned_retron_ids, parse_source_precedent_ids
from .record_ids import parse_record_ids

BENCHLING_GENBANK_DIRNAME = "benchling_genbank"
BENCHLING_ORIENTATION = "reverse_complement_only"


@dataclass(frozen=True)
class BenchlingGenbankImportPlan:
    orientation: str
    expected_count: int
    included_payload_trim_ids: tuple[str, ...]
    assigned_retron_ids: Mapping[str, str]
    record_ids: Mapping[str, str]
    source_precedent_ids: Mapping[str, str]
    descriptions: Mapping[str, str]
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
        return f"{self.record_id_for(variant_id)}.gb"

    def record_id_for(self, variant_id: str) -> str:
        try:
            return self.record_ids[variant_id]
        except KeyError as exc:
            raise RetronMsdCompilerError(f"Retron Benchling import plan has no record id for {variant_id}") from exc

    def source_precedent_id(self, variant_id: str) -> str:
        try:
            return self.source_precedent_ids[variant_id]
        except KeyError as exc:
            raise RetronMsdCompilerError(
                f"Retron Benchling import plan has no source precedent id for {variant_id}"
            ) from exc

    def description_for(self, variant_id: str) -> str:
        try:
            return self.descriptions[variant_id]
        except KeyError as exc:
            raise RetronMsdCompilerError(f"Retron Benchling import plan has no description for {variant_id}") from exc


def parse_benchling_genbank_import_plan(families: Mapping[str, object]) -> BenchlingGenbankImportPlan:
    raw_plan = _require_mapping(families.get("benchling_genbank_import"), "benchling_genbank_import")
    orientation = str(raw_plan.get("orientation") or "").strip()
    if orientation != BENCHLING_ORIENTATION:
        raise RetronMsdCompilerError(
            f"Retron Benchling GenBank import orientation must be {BENCHLING_ORIENTATION!r}, observed {orientation!r}"
        )
    assigned = parse_assigned_retron_ids(_require_mapping(raw_plan.get("assigned_retron_ids"), "assigned_retron_ids"))
    record_ids = parse_record_ids(raw_plan.get("record_ids"))
    if tuple(record_ids) != tuple(assigned):
        raise RetronMsdCompilerError(
            "Retron Benchling GenBank record_ids must match assigned_retron_ids in order: "
            f"{list(record_ids)} != {list(assigned)}"
        )
    expected_files = _require_string_list(raw_plan.get("expected_files"), "expected_files")
    expected_from_ids = tuple(f"{BENCHLING_GENBANK_DIRNAME}/{record_id}.gb" for record_id in record_ids.values())
    if expected_files != expected_from_ids:
        raise RetronMsdCompilerError(
            "Retron Benchling GenBank expected_files must match record_ids in order: "
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
    descriptions = _require_string_mapping(raw_plan.get("descriptions"), "descriptions")
    if tuple(descriptions) != tuple(assigned):
        raise RetronMsdCompilerError(
            "Retron Benchling GenBank descriptions must match assigned_retron_ids in order: "
            f"{list(descriptions)} != {list(assigned)}"
        )
    included = _require_string_list(raw_plan.get("included_payload_trim_ids"), "included_payload_trim_ids")
    if not included:
        raise RetronMsdCompilerError("Retron Benchling GenBank import must include at least one payload trim id")
    return BenchlingGenbankImportPlan(
        orientation=orientation,
        expected_count=expected_count,
        included_payload_trim_ids=included,
        assigned_retron_ids=assigned,
        record_ids=record_ids,
        source_precedent_ids=precedents,
        descriptions=descriptions,
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


def _require_string_mapping(raw: object, label: str) -> Mapping[str, str]:
    if not isinstance(raw, Mapping) or not raw:
        raise RetronMsdCompilerError(f"Retron Benchling GenBank import expected non-empty mapping for {label}")
    parsed: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.strip():
            raise RetronMsdCompilerError(f"Retron Benchling GenBank import has invalid key in {label}")
        if not isinstance(value, str) or not value.strip() or "\n" in value:
            raise RetronMsdCompilerError(f"Retron Benchling GenBank import has invalid value for {label}.{key}")
        parsed[key.strip()] = value.strip()
    return parsed


__all__ = ["BENCHLING_GENBANK_DIRNAME", "BenchlingGenbankImportPlan", "parse_benchling_genbank_import_plan"]
