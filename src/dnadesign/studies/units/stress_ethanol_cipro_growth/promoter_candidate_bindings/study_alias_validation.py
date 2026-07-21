"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/promoter_candidate_bindings/study_alias_validation.py

Validation helpers for the stable promoter alias registry.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import PromoterCandidateBindingsError
from .study_alias_contracts import AliasFirstAssignment, AliasFormat, StudyPromoterAlias, sequence_sha256
from .values import required_text

ASSIGNMENT_FIELDS = {
    "ordinal",
    "alias",
    "candidate_id",
    "sequence_sha256",
    "first_assignment",
    "source_aliases",
}
FIRST_ASSIGNMENT_FIELDS = {
    "source_authority",
    "source_id",
    "nomination_batch_index",
    "model_as_of_round",
}


def mapping(value: object, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PromoterCandidateBindingsError(f"{context} must be a mapping.")
    return {str(key): item for key, item in value.items()}


def alias_format_from(value: object) -> AliasFormat:
    raw = mapping(value, context="format")
    if set(raw) != {"prefix", "zero_pad_width"}:
        raise PromoterCandidateBindingsError("Alias registry format fields must be prefix and zero_pad_width.")
    prefix = required_text(raw["prefix"], field="alias prefix")
    if re.fullmatch(r"[A-Z][A-Z0-9]*", prefix) is None:
        raise PromoterCandidateBindingsError("Alias prefix must contain only uppercase letters and digits.")
    return AliasFormat(
        prefix=prefix,
        zero_pad_width=_positive_integer(raw["zero_pad_width"], field="alias zero-pad width"),
    )


def assignments_from(value: object, *, alias_format: AliasFormat) -> tuple[StudyPromoterAlias, ...]:
    if not isinstance(value, list) or not value:
        raise PromoterCandidateBindingsError("Alias registry assignments must be a non-empty list.")
    rows: list[StudyPromoterAlias] = []
    for index, item in enumerate(value):
        raw = mapping(item, context=f"assignments[{index}]")
        if set(raw) != ASSIGNMENT_FIELDS:
            raise PromoterCandidateBindingsError(f"Alias registry assignments[{index}] fields do not match v1.")
        ordinal = _positive_integer(raw["ordinal"], field=f"assignments[{index}].ordinal")
        alias = required_text(raw["alias"], field=f"assignments[{index}].alias")
        expected_alias = alias_format.render(ordinal)
        if alias != expected_alias:
            raise PromoterCandidateBindingsError(
                f"Alias {alias!r} does not match ordinal {ordinal}; expected {expected_alias!r}."
            )
        digest = required_text(raw["sequence_sha256"], field="sequence SHA-256").lower()
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise PromoterCandidateBindingsError("Alias registry sequence_sha256 must be a lowercase SHA-256 digest.")
        source_alias_values = raw["source_aliases"]
        if not isinstance(source_alias_values, list):
            raise PromoterCandidateBindingsError("Alias registry source_aliases must be a list.")
        source_aliases = tuple(required_text(item, field="source alias") for item in source_alias_values)
        if len(source_aliases) != len(set(source_aliases)):
            raise PromoterCandidateBindingsError(f"Alias {alias!r} contains duplicate source aliases.")
        rows.append(
            StudyPromoterAlias(
                ordinal=ordinal,
                alias=alias,
                candidate_id=required_text(raw["candidate_id"], field="candidate ID"),
                sequence_sha256=digest,
                first_assignment=_first_assignment(raw["first_assignment"], index=index),
                source_aliases=source_aliases,
            )
        )
    _validate_assignment_set(rows)
    return tuple(rows)


def validate_candidate_table(path: Path, *, assignments: tuple[StudyPromoterAlias, ...]) -> None:
    if not path.is_file():
        raise PromoterCandidateBindingsError(f"Alias registry candidate table not found: {path}")
    try:
        records = pd.read_parquet(path, columns=["id", "sequence"])
    except (KeyError, ValueError) as exc:
        raise PromoterCandidateBindingsError(
            "Alias registry candidate table requires id and sequence columns."
        ) from exc
    records = records.copy()
    records["id"] = records["id"].astype(str).str.strip()
    if records["id"].duplicated().any():
        duplicates = sorted(records.loc[records["id"].duplicated(keep=False), "id"].unique().tolist())
        raise PromoterCandidateBindingsError(f"Alias registry candidate table contains duplicate IDs: {duplicates[:5]}")
    by_id = records.set_index("id")["sequence"].astype(str).to_dict()
    for row in assignments:
        sequence = by_id.get(row.candidate_id)
        if sequence is None:
            raise PromoterCandidateBindingsError(
                f"Alias {row.alias} references candidate absent from candidate table: {row.candidate_id}"
            )
        if sequence_sha256(sequence) != row.sequence_sha256:
            raise PromoterCandidateBindingsError(f"Alias {row.alias} sequence digest mismatch against candidate table.")


def _validate_assignment_set(rows: list[StudyPromoterAlias]) -> None:
    ordinals = [row.ordinal for row in rows]
    if ordinals != list(range(1, len(rows) + 1)):
        raise PromoterCandidateBindingsError(
            f"Alias registry ordinals must be contiguous and ordered from 1; observed {ordinals[:10]}."
        )
    _require_unique([row.alias for row in rows], label="aliases")
    _require_unique([row.candidate_id for row in rows], label="candidate IDs")
    _require_unique([row.sequence_sha256 for row in rows], label="sequences")
    all_source_aliases = [alias for row in rows for alias in row.source_aliases]
    _require_unique(all_source_aliases, label="source aliases")
    collisions = sorted(set(all_source_aliases).intersection(row.alias for row in rows))
    if collisions:
        raise PromoterCandidateBindingsError(f"Source aliases collide with canonical aliases: {collisions[:5]}")


def _first_assignment(value: object, *, index: int) -> AliasFirstAssignment:
    raw = mapping(value, context=f"assignments[{index}].first_assignment")
    if set(raw) != FIRST_ASSIGNMENT_FIELDS:
        raise PromoterCandidateBindingsError("Alias registry first_assignment fields do not match v1.")
    return AliasFirstAssignment(
        source_authority=required_text(raw["source_authority"], field="first assignment source authority"),
        source_id=required_text(raw["source_id"], field="first assignment source ID"),
        nomination_batch_index=_optional_nonnegative_integer(
            raw["nomination_batch_index"],
            field="nomination batch index",
        ),
        model_as_of_round=_optional_nonnegative_integer(raw["model_as_of_round"], field="model as-of round"),
    )


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise PromoterCandidateBindingsError(f"{field} must be a positive integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise PromoterCandidateBindingsError(f"{field} must be a positive integer.") from exc
    if parsed <= 0 or parsed != value:
        raise PromoterCandidateBindingsError(f"{field} must be a positive integer.")
    return parsed


def _optional_nonnegative_integer(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise PromoterCandidateBindingsError(f"{field} must be null or a non-negative integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise PromoterCandidateBindingsError(f"{field} must be null or a non-negative integer.") from exc
    if parsed < 0 or parsed != value:
        raise PromoterCandidateBindingsError(f"{field} must be null or a non-negative integer.")
    return parsed


def _require_unique(values: list[str], *, label: str) -> None:
    if len(values) == len(set(values)):
        return
    duplicates = sorted({value for value in values if values.count(value) > 1})
    raise PromoterCandidateBindingsError(f"Alias registry {label} must be unique: {duplicates[:5]}")


__all__ = ["alias_format_from", "assignments_from", "mapping", "validate_candidate_table"]
