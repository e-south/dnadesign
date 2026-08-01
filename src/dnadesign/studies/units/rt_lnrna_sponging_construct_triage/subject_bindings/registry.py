"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/subject_bindings/registry.py

Source-closed registry assembly for RT-lnRNA subject bindings.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from .authorities import STUDY_ID
from .contracts import (
    ResolvedSubjectBinding,
    SubjectBinding,
    SubjectBindingContractError,
    SubjectBindingRegistry,
)
from .projection import project_source_sets
from .sources import SourceCache
from .subjects import parse_subject
from .validation import mapping, object_list, require_exact_fields, text

_SCHEMA_ID = "rt_lnrna_subject_binding_registry_v1"
_ROOT_FIELDS = {"schema_id", "study_id", "binding_set_id", "source_sets", "subjects"}


def assemble_registry(
    *,
    repo_root: Path,
    registry_path: Path,
    require_sequence_bytes: bool,
    selected_subject_ids: frozenset[str] | None = None,
) -> tuple[SubjectBindingRegistry, tuple[ResolvedSubjectBinding, ...]]:
    root = Path(repo_root).expanduser().resolve()
    path = Path(registry_path).expanduser().resolve()
    sources = SourceCache()
    payload = mapping(sources.load_yaml(path), label="registry")
    require_exact_fields(payload, _ROOT_FIELDS, label="registry")
    schema_id = text(payload["schema_id"], label="schema_id")
    study_id = text(payload["study_id"], label="study_id")
    if schema_id != _SCHEMA_ID:
        raise SubjectBindingContractError(f"schema_id must be {_SCHEMA_ID}")
    if study_id != STUDY_ID:
        raise SubjectBindingContractError(f"study_id must be {STUDY_ID}")
    binding_set_id = text(payload["binding_set_id"], label="binding_set_id")
    projected_subjects, excluded_study_variant_ids = project_source_sets(
        root=root,
        sources=sources,
        source_sets=object_list(payload["source_sets"], label="source_sets"),
    )
    all_raw_subjects = (*projected_subjects, *object_list(payload["subjects"], label="subjects"))
    raw_subjects = _select_raw_subjects(all_raw_subjects, selected_subject_ids=selected_subject_ids)
    subjects = []
    resolved_subjects: list[ResolvedSubjectBinding] = []
    subject_ids: set[str] = set()
    study_variant_ids: set[str] = set()
    aliases: dict[tuple[str, str], str] = {}
    component_keys: dict[tuple[str, str, str], str] = {}
    for index, raw_subject in enumerate(raw_subjects):
        subject, lnrna_sequence, rt_cds_sequence = parse_subject(
            root=root,
            sources=sources,
            payload=mapping(raw_subject, label=f"subjects[{index}]"),
            index=index,
            require_sequence_bytes=require_sequence_bytes,
        )
        _require_unique_subject(
            subject=subject,
            subject_ids=subject_ids,
            study_variant_ids=study_variant_ids,
            aliases=aliases,
            component_keys=component_keys,
        )
        subjects.append(subject)
        if require_sequence_bytes:
            if rt_cds_sequence is None:
                raise SubjectBindingContractError(f"{subject.subject_id}: RT CDS bytes are unavailable")
            resolved_subjects.append(
                ResolvedSubjectBinding(
                    binding_set_id=binding_set_id,
                    binding=subject,
                    lnrna_sequence=lnrna_sequence,
                    rt_cds_sequence=rt_cds_sequence,
                )
            )
    if not subjects:
        raise SubjectBindingContractError("subjects must contain at least one binding")
    missing_exclusion_coverage = (
        [] if selected_subject_ids is not None else sorted(excluded_study_variant_ids - study_variant_ids)
    )
    if missing_exclusion_coverage:
        raise SubjectBindingContractError(
            "source-set exclusions require explicit subjects with matching study_variant_id: "
            + ", ".join(missing_exclusion_coverage)
        )
    registry = SubjectBindingRegistry._from_source_closed_subjects(
        schema_id=schema_id,
        study_id=study_id,
        binding_set_id=binding_set_id,
        subjects=tuple(subjects),
    )
    return registry, tuple(resolved_subjects)


def _select_raw_subjects(
    raw_subjects: tuple[object, ...], *, selected_subject_ids: frozenset[str] | None
) -> tuple[object, ...]:
    if selected_subject_ids is None:
        return raw_subjects
    selected = tuple(
        raw_subject
        for raw_subject in raw_subjects
        if mapping(raw_subject, label="subjects[]").get("subject_id") in selected_subject_ids
    )
    observed_ids = {
        text(mapping(raw_subject, label="subjects[]").get("subject_id"), label="subjects[].subject_id")
        for raw_subject in selected
    }
    missing_ids = sorted(selected_subject_ids - observed_ids)
    if missing_ids:
        raise SubjectBindingContractError("unknown selected subject_id(s): " + ", ".join(missing_ids))
    return selected


def _require_unique_subject(
    *,
    subject: SubjectBinding,
    subject_ids: set[str],
    study_variant_ids: set[str],
    aliases: dict[tuple[str, str], str],
    component_keys: dict[tuple[str, str, str], str],
) -> None:
    subject_id = subject.subject_id
    if subject_id in subject_ids:
        raise SubjectBindingContractError(f"duplicate subject_id {subject_id!r}")
    subject_ids.add(subject_id)
    if subject.study_variant_id in study_variant_ids:
        raise SubjectBindingContractError(f"duplicate study_variant_id {subject.study_variant_id!r}")
    study_variant_ids.add(subject.study_variant_id)
    component_key = (
        subject.rt_part.sequence_sha256,
        subject.lnrna_part.sequence_sha256,
        subject.payload_program_id,
    )
    prior_subject = component_keys.get(component_key)
    if prior_subject is not None:
        raise SubjectBindingContractError(
            f"duplicate component binding resolves to both {prior_subject!r} and {subject_id!r}"
        )
    component_keys[component_key] = subject_id
    for alias in subject.aliases:
        key = (alias.namespace, alias.value)
        prior_subject = aliases.get(key)
        if prior_subject is not None:
            raise SubjectBindingContractError(
                f"ambiguous alias {alias.namespace}:{alias.value!r} resolves to both "
                f"{prior_subject!r} and {subject_id!r}"
            )
        aliases[key] = subject_id


__all__ = ["assemble_registry"]
