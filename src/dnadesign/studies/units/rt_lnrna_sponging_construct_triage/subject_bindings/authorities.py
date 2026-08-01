"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/subject_bindings/authorities.py

RT, lnRNA, and MSD source-authority verification.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from Bio.Seq import Seq
from pydantic import ValidationError as PydanticValidationError

from dnadesign.contracts.sequence import RtPartPublicationV1

from .contracts import MsdStructureRef, PartAuthorityRef, SubjectBindingContractError
from .sources import SourceCache, contained_file, source_path
from .validation import digest, mapping, object_list, require_exact_fields, sha256, span, text

STUDY_ID = "rt_lnrna_sponging_construct_triage"
CATALOG_AUTHORITY_KIND = "rt_lnrna_variant_genbank_catalog"
RT_PART_PUBLICATION_KIND = "rt_part_publication_v1"
_RETIRED_MSD_MATERIALIZATION_IDS = frozenset({"reader_" + "spop_msd_structure_panel_v1"})

_PART_FIELDS = {
    "owner_study_id",
    "part_id",
    "authority_kind",
    "source_path",
    "record_id",
    "sequence_sha256",
}
_MSD_FIELDS = {
    "owner_study_id",
    "source_manifest_path",
    "variant_id",
    "sequence_sha256",
    "orientation_in_lnrna",
    "lnrna_span_0",
    "structure_materialization_id",
    "structure_subject_id",
}


def resolve_part(
    *,
    root: Path,
    sources: SourceCache,
    payload: Mapping[str, object],
    label: str,
    component: str,
    require_sequence_bytes: bool,
) -> tuple[PartAuthorityRef, str | None]:
    require_exact_fields(payload, _PART_FIELDS, label=label)
    ref = PartAuthorityRef(
        owner_study_id=text(payload["owner_study_id"], label=f"{label}.owner_study_id"),
        part_id=text(payload["part_id"], label=f"{label}.part_id"),
        authority_kind=text(payload["authority_kind"], label=f"{label}.authority_kind"),
        source_path=text(payload["source_path"], label=f"{label}.source_path"),
        record_id=text(payload["record_id"], label=f"{label}.record_id"),
        sequence_sha256=digest(payload["sequence_sha256"], label=f"{label}.sequence_sha256"),
    )
    authority_path = source_path(root, ref.source_path, label=label)
    if ref.authority_kind == CATALOG_AUTHORITY_KIND:
        observed_digest, sequence = _resolve_catalog_part(
            root=root,
            sources=sources,
            source_path=authority_path,
            ref=ref,
            component=component,
            label=label,
        )
    elif ref.authority_kind == RT_PART_PUBLICATION_KIND and component == "rt_cds":
        ref, observed_digest, sequence = _resolve_published_rt_part(
            sources=sources,
            source_path=authority_path,
            ref=ref,
            label=label,
        )
    else:
        raise SubjectBindingContractError(f"{label}: unsupported authority_kind {ref.authority_kind!r} for {component}")
    if observed_digest != ref.sequence_sha256:
        prefix = "projection blocked: RT CDS authority digest" if component == "rt_cds" else "sequence authority digest"
        raise SubjectBindingContractError(
            f"{label}: {prefix} mismatch; declared {ref.sequence_sha256}, observed {observed_digest}"
        )
    if sequence is None and require_sequence_bytes:
        raise SubjectBindingContractError(
            f"{label}: RT CDS bytes are not published by rt_part_publication_v1; "
            f"provider reference {ref.provider_ref!r} must be resolved through a provider-owned byte authority"
        )
    return ref, sequence


def _resolve_catalog_part(
    *,
    root: Path,
    sources: SourceCache,
    source_path: Path,
    ref: PartAuthorityRef,
    component: str,
    label: str,
) -> tuple[str, str]:
    if ref.owner_study_id != STUDY_ID:
        raise SubjectBindingContractError(
            f"{label}: owner_study_id must be {STUDY_ID!r} for the study-owned GenBank catalog"
        )
    source = mapping(sources.load_yaml(source_path), label=f"{label}.source")
    records = mapping(source.get("records"), label=f"{label}.source.records")
    record = mapping(records.get(ref.record_id), label=f"{label}.source.records.{ref.record_id}")
    authority = mapping(record.get(component), label=f"{label}.{component}")
    observed_id = text(authority.get("sequence_id"), label=f"{label}.{component}.sequence_id")
    observed_digest = digest(authority.get("sequence_sha256"), label=f"{label}.{component}.sequence_sha256")
    if component == "lnrna":
        if ref.part_id != ref.record_id:
            raise SubjectBindingContractError(f"{label}: lnRNA part_id must equal catalog record_id")
    elif observed_id != ref.part_id:
        raise SubjectBindingContractError(f"{label}: RT part_id does not match the catalog sequence_id")
    sequence = extract_catalog_sequence(
        root=root,
        sources=sources,
        catalog=source,
        authority=authority,
        label=label,
    )
    if sha256(sequence) != observed_digest:
        raise SubjectBindingContractError(f"{label}: source GenBank sequence digest disagrees with catalog")
    return observed_digest, sequence


def _resolve_published_rt_part(
    *, sources: SourceCache, source_path: Path, ref: PartAuthorityRef, label: str
) -> tuple[PartAuthorityRef, str, None]:
    try:
        publication = RtPartPublicationV1.model_validate(sources.load_yaml(source_path))
    except PydanticValidationError as exc:
        raise SubjectBindingContractError(f"{label}: invalid RT part publication: {exc}") from exc
    if publication.owner_study_id != ref.owner_study_id:
        raise SubjectBindingContractError(
            f"{label}: publication owner {publication.owner_study_id!r} does not match "
            f"owner_study_id {ref.owner_study_id!r}"
        )
    matches = [part for part in publication.parts if part.part_id == ref.record_id]
    if len(matches) != 1:
        raise SubjectBindingContractError(f"{label}: expected one RT part publication row for {ref.record_id!r}")
    part = matches[0]
    if part.part_id != ref.part_id:
        raise SubjectBindingContractError(f"{label}: part_id does not match the RT part publication row")
    enriched = replace(
        ref,
        provider_ref=part.provider_ref,
        cds_length_nt=part.cds_length_nt,
        terminal_stop_codon=part.terminal_stop_codon,
        protein_sha256=part.protein_sha256,
        protein_length_aa=part.protein_length_aa,
    )
    return enriched, part.cds_sha256, None


def resolve_msd_structure(
    *, root: Path, sources: SourceCache, payload: Mapping[str, object], label: str, lnrna_sequence: str
) -> MsdStructureRef:
    require_exact_fields(payload, _MSD_FIELDS, label=label)
    lnrna_span = span(payload["lnrna_span_0"], label=f"{label}.lnrna_span_0")
    ref = MsdStructureRef(
        owner_study_id=text(payload["owner_study_id"], label=f"{label}.owner_study_id"),
        source_manifest_path=text(payload["source_manifest_path"], label=f"{label}.source_manifest_path"),
        variant_id=text(payload["variant_id"], label=f"{label}.variant_id"),
        sequence_sha256=digest(payload["sequence_sha256"], label=f"{label}.sequence_sha256"),
        orientation_in_lnrna=text(payload["orientation_in_lnrna"], label=f"{label}.orientation_in_lnrna"),
        lnrna_span_0=lnrna_span,
        structure_materialization_id=text(
            payload["structure_materialization_id"], label=f"{label}.structure_materialization_id"
        ),
        structure_subject_id=text(payload["structure_subject_id"], label=f"{label}.structure_subject_id"),
    )
    if ref.owner_study_id != "retron_hairpin_design":
        raise SubjectBindingContractError(f"{label}.owner_study_id must be retron_hairpin_design")
    if ref.structure_materialization_id in _RETIRED_MSD_MATERIALIZATION_IDS:
        raise SubjectBindingContractError(f"{label}: retired structure_materialization_id is not accepted")
    if ref.orientation_in_lnrna not in {"forward", "reverse_complement"}:
        raise SubjectBindingContractError(f"{label}.orientation_in_lnrna is unsupported")
    manifest_path = source_path(root, ref.source_manifest_path, label=label)
    if ref.structure_materialization_id != manifest_path.parent.name:
        raise SubjectBindingContractError(
            f"{label}: structure_materialization_id does not match the source bundle directory"
        )
    manifest = mapping(sources.load_yaml(manifest_path), label=f"{label}.source")
    if manifest.get("contract") != "retron_msd_region_record_bundle_v1":
        raise SubjectBindingContractError(f"{label}: unsupported MSD source contract")
    rows = object_list(manifest.get("records"), label=f"{label}.source.records")
    matches = [row for row in rows if isinstance(row, Mapping) and row.get("variant_id") == ref.variant_id]
    if len(matches) != 1:
        raise SubjectBindingContractError(f"{label}: expected one MSD record for {ref.variant_id!r}")
    manifest_row = mapping(matches[0], label=f"{label}.source.records.{ref.variant_id}")
    record_path = contained_file(
        manifest_path.parent,
        text(manifest_row.get("record"), label=f"{label}.record"),
        label=f"{label}.record",
    )
    record = mapping(sources.load_yaml(record_path), label=f"{label}.record")
    if record.get("contract") != "retron_msd_region_record_v1" or record.get("variant_id") != ref.variant_id:
        raise SubjectBindingContractError(f"{label}: MSD record contract or identity mismatch")
    msd_sequence = text(record.get("msd_sequence_5to3"), label=f"{label}.msd_sequence_5to3").upper()
    observed_digest = digest(record.get("msd_sequence_sha256"), label=f"{label}.msd_sequence_sha256")
    if sha256(msd_sequence) != observed_digest or observed_digest != ref.sequence_sha256:
        raise SubjectBindingContractError(f"{label}: MSD source digest mismatch")
    expected = msd_sequence if ref.orientation_in_lnrna == "forward" else str(Seq(msd_sequence).reverse_complement())
    start, end = lnrna_span
    if end > len(lnrna_sequence) or lnrna_sequence[start:end] != expected:
        raise SubjectBindingContractError(f"{label}: MSD sequence does not match lnRNA span in declared orientation")
    if ref.structure_subject_id != text(record.get("display_id"), label=f"{label}.display_id"):
        raise SubjectBindingContractError(f"{label}: structure_subject_id does not match MSD display_id")
    return ref


def extract_catalog_sequence(
    *,
    root: Path,
    sources: SourceCache,
    catalog: Mapping[str, object],
    authority: Mapping[str, object],
    label: str,
) -> str:
    sequence_id = text(authority.get("sequence_id"), label=f"{label}.sequence_id")
    source_ref, separator, _record_ref = sequence_id.partition("#")
    if not separator or not source_ref.startswith("genbank:"):
        raise SubjectBindingContractError(f"{label}.sequence_id must use genbank:<file>#<record>")
    source_file = source_ref.removeprefix("genbank:")
    if not source_file or Path(source_file).name != source_file:
        raise SubjectBindingContractError(f"{label}.sequence_id must name one GenBank file")
    genbank_dir_ref = text(catalog.get("genbank_dir"), label=f"{label}.catalog.genbank_dir")
    genbank_dir = Path(genbank_dir_ref)
    if genbank_dir.is_absolute() or ".." in genbank_dir.parts:
        raise SubjectBindingContractError(f"{label}.catalog.genbank_dir must be repo-relative without parent traversal")
    genbank_path = contained_file(
        root,
        (genbank_dir / source_file).as_posix(),
        label=f"{label}.genbank_component",
    )
    genbank_sequence = sources.load_genbank_sequence(genbank_path)
    start, end = span(authority.get("span_0"), label=f"{label}.span_0")
    if end > len(genbank_sequence):
        raise SubjectBindingContractError(f"{label}: source span exceeds GenBank record")
    return genbank_sequence[start:end]


__all__ = [
    "CATALOG_AUTHORITY_KIND",
    "STUDY_ID",
    "extract_catalog_sequence",
    "resolve_msd_structure",
    "resolve_part",
]
