"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/subject_bindings/contracts.py

Typed records for compositional RT-lnRNA subject bindings.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

_SOURCE_CLOSURE_TOKEN = object()


class SubjectBindingContractError(ValueError):
    """Raised when a subject binding is ambiguous or lacks source closure."""


@dataclass(frozen=True, slots=True)
class PartAuthorityRef:
    owner_study_id: str
    part_id: str
    authority_kind: str
    source_path: str
    record_id: str
    sequence_sha256: str
    provider_ref: str | None = None
    cds_length_nt: int | None = None
    terminal_stop_codon: Literal["included", "omitted"] | None = None
    protein_sha256: str | None = None
    protein_length_aa: int | None = None


@dataclass(frozen=True, slots=True)
class MsdStructureRef:
    owner_study_id: str
    source_manifest_path: str
    variant_id: str
    sequence_sha256: str
    orientation_in_lnrna: str
    lnrna_span_0: tuple[int, int]
    structure_materialization_id: str
    structure_subject_id: str


@dataclass(frozen=True, slots=True)
class ReaderAlias:
    namespace: str
    value: str


@dataclass(frozen=True, slots=True)
class SubjectBinding:
    subject_id: str
    study_variant_id: str
    payload_program_id: str
    rt_part: PartAuthorityRef
    lnrna_part: PartAuthorityRef
    msd_structure: MsdStructureRef | None
    aliases: tuple[ReaderAlias, ...]
    construct_projection_status: str


@dataclass(frozen=True, slots=True)
class SubjectBindingRegistry:
    schema_id: str
    study_id: str
    binding_set_id: str
    subjects: tuple[SubjectBinding, ...]
    _source_closure: object | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        subject_ids: set[str] = set()
        study_variant_ids: set[str] = set()
        component_keys: set[tuple[str, str, str]] = set()
        aliases: set[tuple[str, str]] = set()
        for subject in self.subjects:
            if subject.subject_id in subject_ids:
                raise SubjectBindingContractError(f"duplicate subject_id {subject.subject_id!r}")
            subject_ids.add(subject.subject_id)
            if subject.study_variant_id in study_variant_ids:
                raise SubjectBindingContractError(f"duplicate study_variant_id {subject.study_variant_id!r}")
            study_variant_ids.add(subject.study_variant_id)
            component_key = (
                subject.rt_part.sequence_sha256,
                subject.lnrna_part.sequence_sha256,
                subject.payload_program_id,
            )
            if component_key in component_keys:
                raise SubjectBindingContractError(
                    f"duplicate component binding includes subject {subject.subject_id!r}"
                )
            component_keys.add(component_key)
            for alias in subject.aliases:
                key = (alias.namespace, alias.value)
                if key in aliases:
                    raise SubjectBindingContractError(f"ambiguous alias {alias.namespace}:{alias.value!r}")
                aliases.add(key)

    @classmethod
    def _from_source_closed_subjects(
        cls,
        *,
        schema_id: str,
        study_id: str,
        binding_set_id: str,
        subjects: tuple[SubjectBinding, ...],
    ) -> "SubjectBindingRegistry":
        registry = cls(
            schema_id=schema_id,
            study_id=study_id,
            binding_set_id=binding_set_id,
            subjects=subjects,
        )
        object.__setattr__(registry, "_source_closure", _SOURCE_CLOSURE_TOKEN)
        return registry

    @property
    def is_source_closed(self) -> bool:
        return self._source_closure is _SOURCE_CLOSURE_TOKEN

    @property
    def subjects_by_id(self) -> dict[str, SubjectBinding]:
        return {subject.subject_id: subject for subject in self.subjects}

    @property
    def subjects_by_alias(self) -> dict[tuple[str, str], SubjectBinding]:
        return {(alias.namespace, alias.value): subject for subject in self.subjects for alias in subject.aliases}

    def resolve_subject_id(self, subject_id: str) -> SubjectBinding:
        """Resolve one exact subject id or fail without normalization."""

        subject = self.subjects_by_id.get(subject_id)
        if subject is None:
            raise SubjectBindingContractError(f"unknown subject_id {subject_id!r}")
        return subject

    def resolve_alias(self, *, namespace: str, value: str) -> SubjectBinding:
        """Resolve one exact namespace-qualified alias or fail."""

        subject = self.subjects_by_alias.get((namespace, value))
        if subject is None:
            raise SubjectBindingContractError(f"unknown alias {namespace}:{value!r}")
        return subject


@dataclass(frozen=True, slots=True)
class ResolvedSubjectBinding:
    """One binding whose owners make both component byte payloads available."""

    binding_set_id: str
    binding: SubjectBinding
    lnrna_sequence: str
    rt_cds_sequence: str


@dataclass(frozen=True, slots=True)
class SubjectBindingByteBlock:
    """One valid subject whose provider publication intentionally omits RT bytes."""

    subject_id: str
    owner_study_id: str
    part_id: str
    provider_ref: str
    cds_sha256: str
    reason: str


@dataclass(frozen=True, slots=True)
class SubjectBindingMaterializationResolution:
    """Independently resolved subjects plus explicit provider byte blocks."""

    resolved_subjects: tuple[ResolvedSubjectBinding, ...]
    blocked_subjects: tuple[SubjectBindingByteBlock, ...]


__all__ = [
    "MsdStructureRef",
    "PartAuthorityRef",
    "ReaderAlias",
    "ResolvedSubjectBinding",
    "SubjectBindingByteBlock",
    "SubjectBindingMaterializationResolution",
    "SubjectBinding",
    "SubjectBindingContractError",
    "SubjectBindingRegistry",
]
