"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/subject_bindings/projection.py

Source-set projection and identity-digest verification.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from Bio.Seq import Seq

from .authorities import CATALOG_AUTHORITY_KIND, STUDY_ID, extract_catalog_sequence
from .contracts import SubjectBindingContractError
from .sources import SourceCache, contained_file, source_path
from .validation import digest, mapping, object_list, require_exact_fields, sha256, text

_SOURCE_SET_FIELDS = {
    "source_set_id",
    "projection_kind",
    "owner_study_id",
    "source_path",
    "projection_sha256",
    "msd_source_manifest_path",
    "exclude_record_ids",
    "default_payload_program_id",
    "payload_program_overrides",
}
_SOURCE_SET_ID = "genbank_catalog_projection_v1"
_GENBANK_CATALOG_ID = "rt_lnrna_sponging_construct_triage_retron_variant_genbank_catalog_v1"


def project_source_sets(
    *, root: Path, sources: SourceCache, source_sets: Sequence[object]
) -> tuple[tuple[Mapping[str, object], ...], set[str]]:
    if len(source_sets) != 1:
        raise SubjectBindingContractError("source_sets must contain exactly one catalog projection")
    payload = mapping(source_sets[0], label="source_sets[0]")
    require_exact_fields(payload, _SOURCE_SET_FIELDS, label="source_sets[0]")
    _verify_source_set_identity(payload)
    source_ref = text(payload["source_path"], label="source_sets[0].source_path")
    catalog_path = source_path(root, source_ref, label="source_sets[0]")
    catalog = mapping(sources.load_yaml(catalog_path), label="source_sets[0].source")
    if catalog.get("catalog_id") != _GENBANK_CATALOG_ID or catalog.get("study_id") != STUDY_ID:
        raise SubjectBindingContractError("source_sets[0]: unsupported GenBank catalog identity")
    if catalog.get("ok") is not True or object_list(catalog.get("errors"), label="source_sets[0].source.errors"):
        raise SubjectBindingContractError("source_sets[0]: GenBank catalog is not valid")
    records = mapping(catalog.get("records"), label="source_sets[0].source.records")
    msd_manifest_ref = text(payload["msd_source_manifest_path"], label="source_sets[0].msd_source_manifest_path")
    msd_manifest_path, msd_records = _load_source_set_msd_records(
        root=root,
        sources=sources,
        source_ref=msd_manifest_ref,
    )
    exclusions = _parse_exclusions(payload=payload, records=records)
    overrides = _parse_overrides(payload=payload, records=records, exclusions=exclusions)
    default_payload_program_id = text(
        payload["default_payload_program_id"], label="source_sets[0].default_payload_program_id"
    )
    projected: list[Mapping[str, object]] = []
    projection_identities: list[dict[str, str]] = []
    for record_id, raw_record in records.items():
        variant_id = text(record_id, label="source_sets[0].source.records key")
        if variant_id in exclusions:
            continue
        projected_subject, identity = _project_catalog_record(
            root=root,
            sources=sources,
            catalog=catalog,
            raw_record=raw_record,
            variant_id=variant_id,
            owner_study_id=STUDY_ID,
            source_ref=source_ref,
            payload_program_id=overrides.get(variant_id, default_payload_program_id),
            msd_manifest_ref=msd_manifest_ref,
            msd_manifest_path=msd_manifest_path,
            msd_records=msd_records,
        )
        projected.append(projected_subject)
        projection_identities.append(identity)
    declared_projection_digest = digest(payload["projection_sha256"], label="source_sets[0].projection_sha256")
    observed_projection_digest = projection_sha256(projection_identities)
    if observed_projection_digest != declared_projection_digest:
        raise SubjectBindingContractError(
            "source_sets[0]: projected identity digest drifted; "
            f"declared {declared_projection_digest}, observed {observed_projection_digest}"
        )
    return tuple(projected), set(exclusions)


def _verify_source_set_identity(payload: Mapping[str, object]) -> None:
    source_set_id = text(payload["source_set_id"], label="source_sets[0].source_set_id")
    if source_set_id != _SOURCE_SET_ID:
        raise SubjectBindingContractError(f"source_sets[0].source_set_id must be {_SOURCE_SET_ID}")
    projection_kind = text(payload["projection_kind"], label="source_sets[0].projection_kind")
    if projection_kind != CATALOG_AUTHORITY_KIND:
        raise SubjectBindingContractError(f"source_sets[0].projection_kind must be {CATALOG_AUTHORITY_KIND}")
    owner_study_id = text(payload["owner_study_id"], label="source_sets[0].owner_study_id")
    if owner_study_id != STUDY_ID:
        raise SubjectBindingContractError(f"source_sets[0].owner_study_id must be {STUDY_ID}")


def _parse_exclusions(*, payload: Mapping[str, object], records: Mapping[str, object]) -> tuple[str, ...]:
    exclusions = tuple(
        text(value, label="source_sets[0].exclude_record_ids[]")
        for value in object_list(payload["exclude_record_ids"], label="source_sets[0].exclude_record_ids")
    )
    if len(set(exclusions)) != len(exclusions):
        raise SubjectBindingContractError("source_sets[0].exclude_record_ids must be unique")
    missing_exclusions = sorted(set(exclusions) - set(records))
    if missing_exclusions:
        raise SubjectBindingContractError(
            "source_sets[0].exclude_record_ids are absent from the catalog: " + ", ".join(missing_exclusions)
        )
    return exclusions


def _parse_overrides(
    *, payload: Mapping[str, object], records: Mapping[str, object], exclusions: tuple[str, ...]
) -> dict[str, str]:
    overrides_payload = mapping(payload["payload_program_overrides"], label="source_sets[0].payload_program_overrides")
    overrides = {
        text(key, label="source_sets[0].payload_program_overrides key"): text(
            value, label=f"source_sets[0].payload_program_overrides.{key}"
        )
        for key, value in overrides_payload.items()
    }
    invalid_overrides = sorted(set(overrides) - (set(records) - set(exclusions)))
    if invalid_overrides:
        raise SubjectBindingContractError(
            "source_sets[0].payload_program_overrides do not name included catalog records: "
            + ", ".join(invalid_overrides)
        )
    return overrides


def _project_catalog_record(
    *,
    root: Path,
    sources: SourceCache,
    catalog: Mapping[str, object],
    raw_record: object,
    variant_id: str,
    owner_study_id: str,
    source_ref: str,
    payload_program_id: str,
    msd_manifest_ref: str,
    msd_manifest_path: Path,
    msd_records: Mapping[str, Mapping[str, object]],
) -> tuple[Mapping[str, object], dict[str, str]]:
    record = mapping(raw_record, label=f"source_sets[0].source.records.{variant_id}")
    if text(record.get("variant_id"), label=f"{variant_id}.variant_id") != variant_id:
        raise SubjectBindingContractError(f"{variant_id}: catalog record key does not match variant_id")
    lnrna = mapping(record.get("lnrna"), label=f"{variant_id}.lnrna")
    rt_cds = mapping(record.get("rt_cds"), label=f"{variant_id}.rt_cds")
    lnrna_sequence = extract_catalog_sequence(
        root=root,
        sources=sources,
        catalog=catalog,
        authority=lnrna,
        label=f"source_sets[0].source.records.{variant_id}.lnrna",
    )
    subject_id = text(record.get("construct_subject_id"), label=f"{variant_id}.construct_subject_id")
    reader_design_id = text(record.get("reader_design_id"), label=f"{variant_id}.reader_design_id")
    rt_sequence_sha256 = digest(rt_cds.get("sequence_sha256"), label=f"{variant_id}.rt_cds.sequence_sha256")
    lnrna_sequence_sha256 = digest(lnrna.get("sequence_sha256"), label=f"{variant_id}.lnrna.sequence_sha256")
    projection_status = text(
        record.get("construct_projection_status"), label=f"{variant_id}.construct_projection_status"
    )
    source_ref_path = text(record.get("source_path"), label=f"{variant_id}.source_path")
    source_sha256 = digest(record.get("source_sha256"), label=f"{variant_id}.source_sha256")
    observed_source_sha256 = sources.load_file_sha256(source_path(root, source_ref_path, label=f"{variant_id}.source"))
    if observed_source_sha256 != source_sha256:
        raise SubjectBindingContractError(
            f"{variant_id}: GenBank source file digest mismatch: "
            f"declared {source_sha256}, observed {observed_source_sha256}"
        )
    source_kind = text(record.get("source_kind"), label=f"{variant_id}.source_kind")
    benchling_url = text(record.get("benchling_url"), label=f"{variant_id}.benchling_url")
    identity = {
        "record_id": variant_id,
        "study_variant_id": variant_id,
        "subject_id": subject_id,
        "reader_design_id": reader_design_id,
        "source_path": source_ref_path,
        "source_sha256": source_sha256,
        "source_kind": source_kind,
        "benchling_url": benchling_url,
        "rt_sequence_sha256": rt_sequence_sha256,
        "lnrna_sequence_sha256": lnrna_sequence_sha256,
        "payload_program_id": payload_program_id,
        "construct_projection_status": projection_status,
    }
    subject = {
        "subject_id": subject_id,
        "study_variant_id": variant_id,
        "payload_program_id": payload_program_id,
        "rt_part": _projected_part_ref(
            owner_study_id=owner_study_id,
            source_path=source_ref,
            record_id=variant_id,
            part_id=text(rt_cds.get("sequence_id"), label=f"{variant_id}.rt_cds.sequence_id"),
            sequence_sha256=rt_sequence_sha256,
        ),
        "lnrna_part": _projected_part_ref(
            owner_study_id=owner_study_id,
            source_path=source_ref,
            record_id=variant_id,
            part_id=variant_id,
            sequence_sha256=lnrna_sequence_sha256,
        ),
        "msd_structure": _projected_msd_structure(
            sources=sources,
            manifest_ref=msd_manifest_ref,
            manifest_path=msd_manifest_path,
            manifest_records=msd_records,
            variant_id=variant_id,
            lnrna_sequence=lnrna_sequence,
            lnrna_sequence_sha256=lnrna_sequence_sha256,
            require_hairpin_source_closure=(benchling_url == "local_retron_hairpin_handoff"),
        ),
        "aliases": [{"namespace": "reader.design_id", "value": reader_design_id}],
        "construct_projection_status": projection_status,
    }
    return subject, identity


def projection_sha256(rows: Sequence[Mapping[str, str]]) -> str:
    canonical_rows = sorted(rows, key=lambda row: (row["study_variant_id"], row["subject_id"]))
    canonical_json = json.dumps(canonical_rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(canonical_json)


def _load_source_set_msd_records(
    *, root: Path, sources: SourceCache, source_ref: str
) -> tuple[Path, dict[str, Mapping[str, object]]]:
    manifest_path = source_path(root, source_ref, label="source_sets[0].msd_source_manifest_path")
    manifest = mapping(sources.load_yaml(manifest_path), label="source_sets[0].msd_source")
    if manifest.get("contract") != "retron_msd_region_record_bundle_v1":
        raise SubjectBindingContractError("source_sets[0]: unsupported MSD source contract")
    indexed: dict[str, Mapping[str, object]] = {}
    for index, raw_record in enumerate(object_list(manifest.get("records"), label="source_sets[0].msd_source.records")):
        record = mapping(raw_record, label=f"source_sets[0].msd_source.records[{index}]")
        variant_id = text(record.get("variant_id"), label=f"source_sets[0].msd_source.records[{index}].variant_id")
        if variant_id in indexed:
            raise SubjectBindingContractError(f"source_sets[0]: duplicate MSD structure record {variant_id!r}")
        indexed[variant_id] = record
    return manifest_path, indexed


def _projected_msd_structure(
    *,
    sources: SourceCache,
    manifest_ref: str,
    manifest_path: Path,
    manifest_records: Mapping[str, Mapping[str, object]],
    variant_id: str,
    lnrna_sequence: str,
    lnrna_sequence_sha256: str,
    require_hairpin_source_closure: bool,
) -> dict[str, object] | None:
    manifest_record = manifest_records.get(variant_id)
    if manifest_record is None:
        if require_hairpin_source_closure:
            raise SubjectBindingContractError(
                f"{variant_id}: local_retron_hairpin_handoff requires an exact hairpin source record"
            )
        return None
    record_path = contained_file(
        manifest_path.parent,
        text(manifest_record.get("record"), label=f"{variant_id}.msd.record"),
        label=f"{variant_id}.msd.record",
    )
    record = mapping(sources.load_yaml(record_path), label=f"{variant_id}.msd.record")
    if record.get("variant_id") != variant_id:
        raise SubjectBindingContractError(f"{variant_id}: MSD record identity mismatch")
    if require_hairpin_source_closure:
        hairpin_source_digest = digest(
            record.get("source_sequence_sha256"),
            label=f"{variant_id}.source_sequence_sha256",
        )
        if hairpin_source_digest != lnrna_sequence_sha256:
            raise SubjectBindingContractError(
                f"{variant_id}: hairpin source sequence digest disagrees with catalog lnRNA digest"
            )
    msd_sequence = text(record.get("msd_sequence_5to3"), label=f"{variant_id}.msd_sequence_5to3").upper()
    matches = _oriented_sequence_matches(lnrna_sequence=lnrna_sequence, msd_sequence=msd_sequence)
    if len(matches) != 1:
        raise SubjectBindingContractError(
            f"{variant_id}: MSD sequence must have one exact forward-or-reverse-complement span in lnRNA; "
            f"found {len(matches)}"
        )
    orientation, start, end = matches[0]
    return {
        "owner_study_id": "retron_hairpin_design",
        "source_manifest_path": manifest_ref,
        "variant_id": variant_id,
        "sequence_sha256": digest(record.get("msd_sequence_sha256"), label=f"{variant_id}.msd_sequence_sha256"),
        "orientation_in_lnrna": orientation,
        "lnrna_span_0": [start, end],
        "structure_materialization_id": manifest_path.parent.name,
        "structure_subject_id": text(record.get("display_id"), label=f"{variant_id}.display_id"),
    }


def _oriented_sequence_matches(*, lnrna_sequence: str, msd_sequence: str) -> tuple[tuple[str, int, int], ...]:
    matches: set[tuple[str, int, int]] = set()
    for orientation, query in (
        ("forward", msd_sequence),
        ("reverse_complement", str(Seq(msd_sequence).reverse_complement())),
    ):
        start = lnrna_sequence.find(query)
        while start >= 0:
            matches.add((orientation, start, start + len(query)))
            start = lnrna_sequence.find(query, start + 1)
    return tuple(sorted(matches, key=lambda item: (item[1], item[2], item[0])))


def _projected_part_ref(
    *, owner_study_id: str, source_path: str, record_id: str, part_id: str, sequence_sha256: str
) -> dict[str, str]:
    return {
        "owner_study_id": owner_study_id,
        "part_id": part_id,
        "authority_kind": CATALOG_AUTHORITY_KIND,
        "source_path": source_path,
        "record_id": record_id,
        "sequence_sha256": sequence_sha256,
    }


__all__ = ["project_source_sets", "projection_sha256"]
