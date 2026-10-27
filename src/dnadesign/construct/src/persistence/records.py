"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/persistence/records.py

Construct output-record persistence contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from dnadesign.usr import SequenceViewRecord

from ..contracts.errors import ValidationError
from .usr_registry import _existing_output_ids


@dataclass
class BuiltRecord:
    output_id: str
    sequence: str
    alphabet: str
    metadata: Dict[str, object]
    label_primary: str | None
    label_aliases: List[str]
    created_at: str
    derived_metadata: Dict[str, object] | None = None
    sequence_view: SequenceViewRecord | None = None


def validate_duplicate_output_aliases(records: list[BuiltRecord]) -> None:
    grouped: dict[str, list[BuiltRecord]] = {}
    for record in records:
        grouped.setdefault(record.output_id, []).append(record)

    invalid: list[str] = []
    for output_id, group in grouped.items():
        if len(group) <= 1:
            continue
        if len({record.sequence for record in group}) != 1:
            invalid.append(output_id)
            continue
        if any(record.sequence_view is None for record in group):
            invalid.append(output_id)
    if invalid:
        preview = ", ".join(sorted(invalid)[:5])
        raise ValidationError(
            f"{len(invalid)} duplicate planned output id(s) were generated within this construct run without "
            f"sequence-view alias coverage. Sample: {preview}. Deduplicate input.ids or route colliding outputs "
            "into separate construct jobs."
        )


def unique_records_by_output_id(records: list[BuiltRecord]) -> list[BuiltRecord]:
    unique: dict[str, BuiltRecord] = {}
    for record in records:
        current = unique.get(record.output_id)
        if current is None:
            unique[record.output_id] = record
            continue
        if current.sequence != record.sequence:
            raise ValidationError(f"Construct output id collision has different sequence payload: {record.output_id}.")
    return list(unique.values())


def ambiguous_row_overlay_ids(records: list[BuiltRecord]) -> set[str]:
    grouped: dict[str, list[BuiltRecord]] = {}
    for record in records:
        grouped.setdefault(record.output_id, []).append(record)

    ambiguous: set[str] = set()
    for output_id, group in grouped.items():
        if len(group) <= 1:
            continue
        first = group[0]
        first_payload = (
            first.metadata,
            first.derived_metadata,
            first.label_primary,
            tuple(first.label_aliases),
        )
        for candidate in group[1:]:
            candidate_payload = (
                candidate.metadata,
                candidate.derived_metadata,
                candidate.label_primary,
                tuple(candidate.label_aliases),
            )
            if candidate_payload != first_payload:
                ambiguous.add(output_id)
                break
    return ambiguous


def output_records_for_overlay(records: list[BuiltRecord]) -> list[BuiltRecord]:
    unique_records = unique_records_by_output_id(records)
    ambiguous_overlay_ids = ambiguous_row_overlay_ids(records)
    return [record for record in unique_records if record.output_id not in ambiguous_overlay_ids]


def require_output_conflict_policy(
    records: list[BuiltRecord],
    *,
    output_root: Path,
    output_dataset: str,
    on_conflict: str,
) -> int:
    existing_ids = _existing_output_ids(output_root, output_dataset)
    collision_count = sum(1 for output_id in {record.output_id for record in records} if output_id in existing_ids)
    if collision_count and on_conflict == "error":
        raise ValidationError(
            f"{collision_count} planned output id(s) already exist in dataset '{output_dataset}'. "
            "Choose a different output dataset, change the construct spec, or set output.on_conflict='ignore'."
        )
    return collision_count


def records_to_write(
    records: list[BuiltRecord],
    *,
    output_root: Path,
    output_dataset: str,
    on_conflict: str,
) -> list[BuiltRecord]:
    existing_ids = _existing_output_ids(output_root, output_dataset)
    return [record for record in records if on_conflict != "ignore" or record.output_id not in existing_ids]
