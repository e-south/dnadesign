"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/subject_bindings/subjects.py

Strict subject and alias parsing over verified component authorities.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from .authorities import STUDY_ID, resolve_msd_structure, resolve_part
from .contracts import PartAuthorityRef, ReaderAlias, SubjectBinding, SubjectBindingContractError
from .sources import SourceCache
from .validation import mapping, object_list, require_exact_fields, text

_PROJECTION_STATUSES = {"representable"}
_ALIAS_NAMESPACES = {"reader.design_id", "reader.assay_subject_id"}
_SUBJECT_FIELDS = {
    "subject_id",
    "study_variant_id",
    "payload_program_id",
    "rt_part",
    "lnrna_part",
    "msd_structure",
    "aliases",
    "construct_projection_status",
}
_ALIAS_FIELDS = {"namespace", "value"}
_EXTENDED_RETRON_DESIGN_ID = re.compile(r"^pES-retron-(\d+)-")


def parse_subject(
    *,
    root: Path,
    sources: SourceCache,
    payload: Mapping[str, object],
    index: int,
    require_sequence_bytes: bool,
) -> tuple[SubjectBinding, str, str | None]:
    label = f"subjects[{index}]"
    require_exact_fields(payload, _SUBJECT_FIELDS, label=label)
    subject_id = text(payload["subject_id"], label=f"{label}.subject_id")
    projection_status = text(payload["construct_projection_status"], label=f"{label}.construct_projection_status")
    if projection_status not in _PROJECTION_STATUSES:
        raise SubjectBindingContractError(
            f"{label}.construct_projection_status must be one of {sorted(_PROJECTION_STATUSES)}"
        )
    rt_part, rt_cds_sequence = resolve_part(
        root=root,
        sources=sources,
        payload=mapping(payload["rt_part"], label=f"{label}.rt_part"),
        label=f"{label}.rt_part",
        component="rt_cds",
        require_sequence_bytes=require_sequence_bytes,
    )
    lnrna_part, lnrna_sequence = resolve_part(
        root=root,
        sources=sources,
        payload=mapping(payload["lnrna_part"], label=f"{label}.lnrna_part"),
        label=f"{label}.lnrna_part",
        component="lnrna",
        require_sequence_bytes=require_sequence_bytes,
    )
    if lnrna_sequence is None:
        raise SubjectBindingContractError(f"{label}.lnrna_part: lnRNA sequence bytes are unavailable")
    raw_msd_structure = payload["msd_structure"]
    msd_structure = (
        None
        if raw_msd_structure is None
        else resolve_msd_structure(
            root=root,
            sources=sources,
            payload=mapping(raw_msd_structure, label=f"{label}.msd_structure"),
            label=f"{label}.msd_structure",
            lnrna_sequence=lnrna_sequence,
        )
    )
    aliases = tuple(
        parse_alias(mapping(item, label=f"{label}.aliases[{alias_index}]"), label=f"{label}.aliases[{alias_index}]")
        for alias_index, item in enumerate(object_list(payload["aliases"], label=f"{label}.aliases"))
    )
    if len({(alias.namespace, alias.value) for alias in aliases}) != len(aliases):
        raise SubjectBindingContractError(f"{subject_id}: duplicate alias within subject")
    study_variant_id = text(payload["study_variant_id"], label=f"{label}.study_variant_id")
    _reject_construct_number_as_composite_identity(
        subject_id=subject_id,
        study_variant_id=study_variant_id,
        rt_part=rt_part,
        aliases=aliases,
    )
    binding = SubjectBinding(
        subject_id=subject_id,
        study_variant_id=study_variant_id,
        payload_program_id=text(payload["payload_program_id"], label=f"{label}.payload_program_id"),
        rt_part=rt_part,
        lnrna_part=lnrna_part,
        msd_structure=msd_structure,
        aliases=aliases,
        construct_projection_status=projection_status,
    )
    return binding, lnrna_sequence, rt_cds_sequence


def parse_alias(payload: Mapping[str, object], *, label: str) -> ReaderAlias:
    require_exact_fields(payload, _ALIAS_FIELDS, label=label)
    namespace = text(payload["namespace"], label=f"{label}.namespace")
    if namespace not in _ALIAS_NAMESPACES:
        raise SubjectBindingContractError(f"{label}.namespace must be one of {sorted(_ALIAS_NAMESPACES)}")
    return ReaderAlias(namespace=namespace, value=text(payload["value"], label=f"{label}.value"))


def _reject_construct_number_as_composite_identity(
    *, subject_id: str, study_variant_id: str, rt_part: PartAuthorityRef, aliases: Sequence[ReaderAlias]
) -> None:
    if rt_part.owner_study_id == STUDY_ID:
        return
    for alias in aliases:
        if alias.namespace != "reader.design_id":
            continue
        match = _EXTENDED_RETRON_DESIGN_ID.match(alias.value)
        if match is None:
            continue
        collapsed_id = f"retron{match.group(1)}"
        if study_variant_id == collapsed_id:
            raise SubjectBindingContractError(
                f"{subject_id}: composite Reader design {alias.value!r} cannot use bare construct-number "
                f"study_variant_id {study_variant_id!r}; use a component-defined identity"
            )


__all__ = ["parse_alias", "parse_subject"]
