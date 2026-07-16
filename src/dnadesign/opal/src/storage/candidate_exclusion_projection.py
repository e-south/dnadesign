"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/storage/candidate_exclusion_projection.py

Bind a study-published candidate-exclusion set to campaign eligibility.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from ..core.utils import OpalError

PROJECTION_FIELDS = {"exclusion_set_id", "entries_sha256", "entry_count"}


@dataclass(frozen=True)
class CandidateExclusionSetBinding:
    """One configured candidate-ID exclusion set presented for verification."""

    exclusion_set_id: str
    entries: Sequence[Mapping[str, object]]


@dataclass(frozen=True)
class VerifiedCandidateExclusionProjection:
    exclusion_set_id: str
    entries_sha256: str
    entry_count: int


def candidate_exclusion_entries_sha256(entries: object) -> str:
    normalized = normalize_candidate_exclusion_entries(entries)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def build_candidate_exclusion_projection(
    *,
    exclusion_set_id: str,
    entries: object,
) -> dict[str, object]:
    normalized = normalize_candidate_exclusion_entries(entries)
    return {
        "exclusion_set_id": _nonempty(exclusion_set_id, field="exclusion_set_id"),
        "entries_sha256": candidate_exclusion_entries_sha256(normalized),
        "entry_count": len(normalized),
    }


def candidate_exclusion_sets_from_config(config: Any) -> tuple[CandidateExclusionSetBinding, ...]:
    """Project configured candidate-ID rules into the immutable-label binding."""

    bindings: list[CandidateExclusionSetBinding] = []
    for rule in config.candidate_eligibility.rules:
        if str(rule.name) != "candidate_id_exclusion":
            continue
        bindings.append(
            CandidateExclusionSetBinding(
                exclusion_set_id=str(rule.params.get("exclusion_set_id", "")),
                entries=rule.params.get("entries"),
            )
        )
    return tuple(bindings)


def verify_candidate_exclusion_projection(
    value: object,
    *,
    configured_sets: Sequence[CandidateExclusionSetBinding] | None,
) -> VerifiedCandidateExclusionProjection:
    if not isinstance(value, Mapping) or set(value) != PROJECTION_FIELDS:
        raise OpalError(
            "Observed-label promotion candidate_exclusion_projection fields must be exactly "
            f"{sorted(PROJECTION_FIELDS)}."
        )
    exclusion_set_id = _nonempty(value.get("exclusion_set_id"), field="exclusion_set_id")
    entries_sha256 = _digest(value.get("entries_sha256"))
    entry_count = value.get("entry_count")
    if isinstance(entry_count, bool) or not isinstance(entry_count, int) or entry_count < 0:
        raise OpalError(
            "Observed-label promotion candidate_exclusion_projection.entry_count must be a non-negative integer."
        )
    if configured_sets is not None:
        _require_configured_projection(
            exclusion_set_id=exclusion_set_id,
            entries_sha256=entries_sha256,
            entry_count=entry_count,
            configured_sets=configured_sets,
        )
    return VerifiedCandidateExclusionProjection(
        exclusion_set_id=exclusion_set_id,
        entries_sha256=entries_sha256,
        entry_count=entry_count,
    )


def normalize_candidate_exclusion_entries(value: object) -> list[dict[str, str]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise OpalError("Candidate exclusion entries must be a list.")
    entries: list[dict[str, str]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != {"candidate_id", "reason"}:
            raise OpalError(f"Candidate exclusion entry {index} must contain exactly candidate_id and reason.")
        entries.append(
            {
                "candidate_id": _nonempty(raw["candidate_id"], field="candidate_id"),
                "reason": _nonempty(raw["reason"], field="reason"),
            }
        )
    candidate_ids = [entry["candidate_id"] for entry in entries]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise OpalError("Candidate exclusion entries contain duplicate candidate IDs.")
    return sorted(entries, key=lambda entry: entry["candidate_id"])


def _require_configured_projection(
    *,
    exclusion_set_id: str,
    entries_sha256: str,
    entry_count: int,
    configured_sets: Sequence[CandidateExclusionSetBinding],
) -> None:
    by_id: dict[str, list[dict[str, str]]] = {}
    for binding in configured_sets:
        binding_id = _nonempty(binding.exclusion_set_id, field="configured exclusion_set_id")
        if binding_id in by_id:
            raise OpalError(f"Campaign declares candidate exclusion set {binding_id!r} more than once.")
        by_id[binding_id] = normalize_candidate_exclusion_entries(binding.entries)
    configured = by_id.get(exclusion_set_id)
    if configured is None:
        if entry_count:
            raise OpalError(f"Campaign is missing candidate exclusion set {exclusion_set_id!r}.")
        configured = []
    if len(configured) != entry_count:
        raise OpalError(
            f"Campaign candidate exclusion entry-count mismatch: expected {entry_count}, found {len(configured)}."
        )
    actual_sha256 = candidate_exclusion_entries_sha256(configured)
    if actual_sha256 != entries_sha256:
        raise OpalError(
            f"Campaign candidate exclusion digest mismatch: expected {entries_sha256}, found {actual_sha256}."
        )


def _nonempty(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise OpalError(f"Candidate exclusion {field} must be canonical non-empty text.")
    return value


def _digest(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise OpalError(
            "Observed-label promotion candidate_exclusion_projection.entries_sha256 must be a lowercase SHA-256 digest."
        )
    return value


__all__ = [
    "CandidateExclusionSetBinding",
    "VerifiedCandidateExclusionProjection",
    "build_candidate_exclusion_projection",
    "candidate_exclusion_entries_sha256",
    "candidate_exclusion_sets_from_config",
    "normalize_candidate_exclusion_entries",
    "verify_candidate_exclusion_projection",
]
