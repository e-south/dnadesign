"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/promoter_candidate_bindings/study_alias_registry.py

Append-only promoter aliases for the stress study.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import yaml

from .contracts import STUDY_ID, PromoterCandidateBindingsError
from .source_registry import relative_config_path
from .study_alias_contracts import (
    AliasFirstAssignment,
    AliasFormat,
    PlannedStudyAlias,
    StudyPromoterAlias,
    StudyPromoterAliasRegistry,
    sequence_sha256,
)
from .study_alias_validation import alias_format_from, assignments_from, mapping, validate_candidate_table
from .values import required_text

REGISTRY_SCHEMA_ID = "dnadesign.study.promoter_alias_registry.v1"
REGISTRY_SCHEMA_VERSION = "1"
REGISTRY_PATH = Path("docs/studies/stress_ethanol_cipro_growth/record/promoter_aliases.yaml")
STUDY_ALIAS_NAMESPACE = "study.promoter_alias"

_ROOT_FIELDS = {
    "schema_id",
    "schema_version",
    "study_id",
    "alias_namespace",
    "format",
    "candidate_table",
    "assignments",
}


def load_study_promoter_alias_registry(
    repo_root: str | Path,
    *,
    registry_path: str | Path = REGISTRY_PATH,
) -> StudyPromoterAliasRegistry:
    """Load and verify the study's exact promoter alias assignments."""

    root = Path(repo_root).expanduser().resolve()
    path = _resolve_registry_path(root, registry_path)
    if not path.is_file():
        raise PromoterCandidateBindingsError(f"Study promoter alias registry not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PromoterCandidateBindingsError(f"Could not parse study promoter alias registry: {exc}") from exc
    payload = mapping(raw, context="study promoter alias registry")
    if set(payload) != _ROOT_FIELDS:
        raise PromoterCandidateBindingsError("Study promoter alias registry fields do not match v1.")
    if (
        payload["schema_id"] != REGISTRY_SCHEMA_ID
        or str(payload["schema_version"]) != REGISTRY_SCHEMA_VERSION
        or payload["study_id"] != STUDY_ID
        or payload["alias_namespace"] != STUDY_ALIAS_NAMESPACE
    ):
        raise PromoterCandidateBindingsError("Study promoter alias registry identity mismatch.")

    alias_format = alias_format_from(payload["format"])
    candidate_table = mapping(payload["candidate_table"], context="candidate_table")
    if set(candidate_table) != {"dataset_id", "records_path"}:
        raise PromoterCandidateBindingsError(
            "Alias registry candidate_table fields must be dataset_id and records_path."
        )
    candidate_table_path = relative_config_path(
        candidate_table["records_path"],
        context="candidate_table.records_path",
    )
    assignments = assignments_from(payload["assignments"], alias_format=alias_format)
    validate_candidate_table(root / candidate_table_path, assignments=assignments)
    return StudyPromoterAliasRegistry(
        path=path.relative_to(root),
        candidate_table_dataset_id=required_text(
            candidate_table["dataset_id"],
            field="candidate table dataset ID",
        ),
        candidate_table_records_path=candidate_table_path,
        alias_format=alias_format,
        assignments=assignments,
    )


def plan_study_aliases(
    registry: StudyPromoterAliasRegistry,
    candidates: Sequence[tuple[str, str]],
) -> tuple[PlannedStudyAlias, ...]:
    """Plan deterministic reuse or append-only aliases without writing the registry."""

    if not candidates:
        raise PromoterCandidateBindingsError("Alias planning requires at least one candidate.")
    registered_by_id = {row.candidate_id: row for row in registry.assignments}
    registered_by_sequence = {row.sequence_sha256: row for row in registry.assignments}
    seen_ids: set[str] = set()
    seen_digests: set[str] = set()
    next_ordinal = registry.next_ordinal
    planned: list[PlannedStudyAlias] = []
    for raw_id, raw_sequence in candidates:
        candidate_id = required_text(raw_id, field="candidate ID")
        sequence = required_text(raw_sequence, field="candidate sequence").upper()
        digest = sequence_sha256(sequence)
        if candidate_id in seen_ids:
            raise PromoterCandidateBindingsError(f"Candidate {candidate_id!r} appears more than once in alias plan.")
        if digest in seen_digests:
            raise PromoterCandidateBindingsError("Candidate sequence appears more than once in alias plan.")
        seen_ids.add(candidate_id)
        seen_digests.add(digest)
        existing = registered_by_id.get(candidate_id)
        if existing is not None:
            if existing.sequence_sha256 != digest:
                raise PromoterCandidateBindingsError(
                    f"Candidate {candidate_id!r} sequence does not match assigned alias {existing.alias}."
                )
            planned.append(
                PlannedStudyAlias(
                    candidate_id=candidate_id,
                    sequence_sha256=digest,
                    alias=existing.alias,
                    ordinal=existing.ordinal,
                    is_new=False,
                )
            )
            continue
        sequence_owner = registered_by_sequence.get(digest)
        if sequence_owner is not None:
            raise PromoterCandidateBindingsError(
                f"Candidate sequence is already assigned to {sequence_owner.alias} ({sequence_owner.candidate_id})."
            )
        planned.append(
            PlannedStudyAlias(
                candidate_id=candidate_id,
                sequence_sha256=digest,
                alias=registry.alias_format.render(next_ordinal),
                ordinal=next_ordinal,
                is_new=True,
            )
        )
        next_ordinal += 1
    return tuple(planned)


def _resolve_registry_path(root: Path, value: str | Path) -> Path:
    raw = Path(value).expanduser()
    path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PromoterCandidateBindingsError(
            "Study promoter alias registry must remain inside the repository."
        ) from exc
    return path


__all__ = [
    "REGISTRY_PATH",
    "REGISTRY_SCHEMA_ID",
    "REGISTRY_SCHEMA_VERSION",
    "STUDY_ALIAS_NAMESPACE",
    "AliasFirstAssignment",
    "AliasFormat",
    "PlannedStudyAlias",
    "StudyPromoterAlias",
    "StudyPromoterAliasRegistry",
    "load_study_promoter_alias_registry",
    "plan_study_aliases",
]
