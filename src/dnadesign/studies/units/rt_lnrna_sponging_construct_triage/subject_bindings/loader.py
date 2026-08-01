"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/subject_bindings/loader.py

Strict source-resolving loader for compositional RT-lnRNA bindings.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import yaml
from Bio import SeqIO
from Bio.Seq import Seq
from pydantic import ValidationError as PydanticValidationError

from dnadesign.contracts.sequence import RtPartPublicationV1

from .contracts import (
    MsdStructureRef,
    PartAuthorityRef,
    ReaderAlias,
    ResolvedSubjectBinding,
    SubjectBinding,
    SubjectBindingByteBlock,
    SubjectBindingContractError,
    SubjectBindingMaterializationResolution,
    SubjectBindingRegistry,
)

_STUDY_DIR = Path("docs/studies/rt_lnrna_sponging_construct_triage")
_DEFAULT_REGISTRY_PATH = _STUDY_DIR / "workbench/provenance/subject_bindings/retron_subject_bindings_v1.yaml"
_SCHEMA_ID = "rt_lnrna_subject_binding_registry_v1"
_STUDY_ID = "rt_lnrna_sponging_construct_triage"
_PROJECTION_STATUSES = {"representable"}
_ALIAS_NAMESPACES = {"reader.design_id", "reader.assay_subject_id"}

_ROOT_FIELDS = {"schema_id", "study_id", "binding_set_id", "source_sets", "subjects"}
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
_SOURCE_SET_PROJECTION_KIND = "rt_lnrna_variant_genbank_catalog"
_GENBANK_CATALOG_ID = "rt_lnrna_sponging_construct_triage_retron_variant_genbank_catalog_v1"
_PART_FIELDS = {
    "owner_study_id",
    "part_id",
    "authority_kind",
    "source_path",
    "record_id",
    "sequence_sha256",
}
_RT_PART_PUBLICATION_KIND = "rt_part_publication_v1"
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
_ALIAS_FIELDS = {"namespace", "value"}
_EXTENDED_RETRON_DESIGN_ID = re.compile(r"^pES-retron-(\d+)-")


class _SourceCache:
    """Reuse immutable source documents within one fail-fast registry load."""

    def __init__(self) -> None:
        self._yaml_by_path: dict[Path, object] = {}
        self._genbank_sequence_by_path: dict[Path, str] = {}
        self._file_sha256_by_path: dict[Path, str] = {}

    def load_yaml(self, path: Path) -> object:
        resolved = path.resolve()
        if resolved not in self._yaml_by_path:
            self._yaml_by_path[resolved] = _load_yaml(resolved)
        return self._yaml_by_path[resolved]

    def load_genbank_sequence(self, path: Path) -> str:
        resolved = path.resolve()
        if resolved not in self._genbank_sequence_by_path:
            self._genbank_sequence_by_path[resolved] = str(SeqIO.read(resolved, "genbank").seq).upper()
        return self._genbank_sequence_by_path[resolved]

    def load_file_sha256(self, path: Path) -> str:
        resolved = path.resolve()
        if resolved not in self._file_sha256_by_path:
            self._file_sha256_by_path[resolved] = f"sha256:{hashlib.sha256(resolved.read_bytes()).hexdigest()}"
        return self._file_sha256_by_path[resolved]


def load_registered_subject_bindings(*, repo_root: Path | None = None) -> SubjectBindingRegistry:
    root = _resolve_repo_root(repo_root)
    return load_subject_bindings(repo_root=root, registry_path=root / _DEFAULT_REGISTRY_PATH)


def load_resolved_registered_subject_bindings(*, repo_root: Path | None = None) -> tuple[ResolvedSubjectBinding, ...]:
    """Load bindings only when every source owner makes exact sequence bytes available."""

    root = _resolve_repo_root(repo_root)
    return load_resolved_subject_bindings(repo_root=root, registry_path=root / _DEFAULT_REGISTRY_PATH)


def load_registered_subject_binding_materialization(
    *,
    repo_root: Path | None = None,
    subject_ids: tuple[str, ...] | None = None,
) -> SubjectBindingMaterializationResolution:
    """Resolve available bytes independently and report opaque provider blocks."""

    root = _resolve_repo_root(repo_root)
    registry_path = root / _DEFAULT_REGISTRY_PATH
    registry = load_subject_bindings(repo_root=root, registry_path=registry_path)
    selected = _selected_subjects(registry=registry, subject_ids=subject_ids)
    blocked = tuple(
        SubjectBindingByteBlock(
            subject_id=subject.subject_id,
            owner_study_id=subject.rt_part.owner_study_id,
            part_id=subject.rt_part.part_id,
            provider_ref=subject.rt_part.provider_ref or "",
            cds_sha256=subject.rt_part.sequence_sha256,
            reason="provider_publication_omits_rt_cds_bytes",
        )
        for subject in selected
        if subject.rt_part.provider_ref is not None
    )
    if subject_ids is not None and blocked:
        details = ", ".join(f"{item.subject_id} ({item.provider_ref})" for item in blocked)
        raise SubjectBindingContractError(f"exact subject projection is byte-blocked: {details}")
    resolvable_ids = frozenset(subject.subject_id for subject in selected if subject.rt_part.provider_ref is None)
    if not resolvable_ids:
        resolved: tuple[ResolvedSubjectBinding, ...] = ()
    else:
        _registry, resolved = _load_subject_bindings(
            repo_root=root,
            registry_path=registry_path,
            require_sequence_bytes=True,
            selected_subject_ids=resolvable_ids,
        )
    return SubjectBindingMaterializationResolution(
        resolved_subjects=resolved,
        blocked_subjects=blocked,
    )


def load_subject_bindings(*, repo_root: Path, registry_path: Path) -> SubjectBindingRegistry:
    registry, _resolved = _load_subject_bindings(
        repo_root=repo_root,
        registry_path=registry_path,
        require_sequence_bytes=False,
    )
    return registry


def load_resolved_subject_bindings(*, repo_root: Path, registry_path: Path) -> tuple[ResolvedSubjectBinding, ...]:
    """Resolve exact bytes or fail when a provider publishes opaque metadata only."""

    _registry, resolved = _load_subject_bindings(
        repo_root=repo_root,
        registry_path=registry_path,
        require_sequence_bytes=True,
    )
    return resolved


def _selected_subjects(
    *,
    registry: SubjectBindingRegistry,
    subject_ids: tuple[str, ...] | None,
) -> tuple[SubjectBinding, ...]:
    if subject_ids is None:
        return registry.subjects
    if not subject_ids:
        raise SubjectBindingContractError("subject_ids must contain at least one exact subject id")
    if len(set(subject_ids)) != len(subject_ids):
        raise SubjectBindingContractError("subject_ids must not contain duplicates")
    selected: list[SubjectBinding] = []
    for subject_id in subject_ids:
        selected.append(registry.resolve_subject_id(subject_id))
    return tuple(selected)


def _load_subject_bindings(
    *,
    repo_root: Path,
    registry_path: Path,
    require_sequence_bytes: bool,
    selected_subject_ids: frozenset[str] | None = None,
) -> tuple[SubjectBindingRegistry, tuple[ResolvedSubjectBinding, ...]]:
    root = Path(repo_root).expanduser().resolve()
    path = Path(registry_path).expanduser().resolve()
    sources = _SourceCache()
    payload = _mapping(sources.load_yaml(path), label="registry")
    _require_exact_fields(payload, _ROOT_FIELDS, label="registry")
    schema_id = _text(payload["schema_id"], label="schema_id")
    study_id = _text(payload["study_id"], label="study_id")
    if schema_id != _SCHEMA_ID:
        raise SubjectBindingContractError(f"schema_id must be {_SCHEMA_ID}")
    if study_id != _STUDY_ID:
        raise SubjectBindingContractError(f"study_id must be {_STUDY_ID}")
    binding_set_id = _text(payload["binding_set_id"], label="binding_set_id")

    subjects: list[SubjectBinding] = []
    resolved_subjects: list[ResolvedSubjectBinding] = []
    subject_ids: set[str] = set()
    study_variant_ids: set[str] = set()
    aliases: dict[tuple[str, str], str] = {}
    component_keys: dict[tuple[str, str, str], str] = {}
    projected_subjects, excluded_study_variant_ids = _project_source_sets(
        root=root,
        sources=sources,
        source_sets=_list(payload["source_sets"], label="source_sets"),
    )
    all_raw_subjects = (*projected_subjects, *_list(payload["subjects"], label="subjects"))
    raw_subjects = (
        all_raw_subjects
        if selected_subject_ids is None
        else tuple(
            raw_subject
            for raw_subject in all_raw_subjects
            if _mapping(raw_subject, label="subjects[]").get("subject_id") in selected_subject_ids
        )
    )
    if selected_subject_ids is not None:
        observed_ids = {
            _text(_mapping(raw_subject, label="subjects[]").get("subject_id"), label="subjects[].subject_id")
            for raw_subject in raw_subjects
        }
        missing_ids = sorted(selected_subject_ids - observed_ids)
        if missing_ids:
            raise SubjectBindingContractError("unknown selected subject_id(s): " + ", ".join(missing_ids))
    for index, raw_subject in enumerate(raw_subjects):
        subject, lnrna_sequence, rt_cds_sequence = _parse_subject(
            root=root,
            sources=sources,
            payload=_mapping(raw_subject, label=f"subjects[{index}]"),
            index=index,
            require_sequence_bytes=require_sequence_bytes,
        )
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
        prior_subject = component_keys.get(component_key)
        if prior_subject is not None:
            raise SubjectBindingContractError(
                f"duplicate component binding resolves to both {prior_subject!r} and {subject.subject_id!r}"
            )
        component_keys[component_key] = subject.subject_id
        for alias in subject.aliases:
            key = (alias.namespace, alias.value)
            prior_subject = aliases.get(key)
            if prior_subject is not None:
                raise SubjectBindingContractError(
                    f"ambiguous alias {alias.namespace}:{alias.value!r} resolves to both "
                    f"{prior_subject!r} and {subject.subject_id!r}"
                )
            aliases[key] = subject.subject_id
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


def _parse_subject(
    *,
    root: Path,
    sources: _SourceCache,
    payload: Mapping[str, object],
    index: int,
    require_sequence_bytes: bool,
) -> tuple[SubjectBinding, str, str | None]:
    label = f"subjects[{index}]"
    _require_exact_fields(payload, _SUBJECT_FIELDS, label=label)
    subject_id = _text(payload["subject_id"], label=f"{label}.subject_id")
    projection_status = _text(payload["construct_projection_status"], label=f"{label}.construct_projection_status")
    if projection_status not in _PROJECTION_STATUSES:
        raise SubjectBindingContractError(
            f"{label}.construct_projection_status must be one of {sorted(_PROJECTION_STATUSES)}"
        )
    rt_part, rt_cds_sequence = _resolve_part(
        root=root,
        sources=sources,
        payload=_mapping(payload["rt_part"], label=f"{label}.rt_part"),
        label=f"{label}.rt_part",
        component="rt_cds",
        require_sequence_bytes=require_sequence_bytes,
    )
    lnrna_part, lnrna_sequence = _resolve_part(
        root=root,
        sources=sources,
        payload=_mapping(payload["lnrna_part"], label=f"{label}.lnrna_part"),
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
        else _resolve_msd_structure(
            root=root,
            sources=sources,
            payload=_mapping(raw_msd_structure, label=f"{label}.msd_structure"),
            label=f"{label}.msd_structure",
            lnrna_sequence=lnrna_sequence,
        )
    )
    aliases = tuple(
        _parse_alias(_mapping(item, label=f"{label}.aliases[{alias_index}]"), label=f"{label}.aliases[{alias_index}]")
        for alias_index, item in enumerate(_list(payload["aliases"], label=f"{label}.aliases"))
    )
    if len({(alias.namespace, alias.value) for alias in aliases}) != len(aliases):
        raise SubjectBindingContractError(f"{subject_id}: duplicate alias within subject")
    study_variant_id = _text(payload["study_variant_id"], label=f"{label}.study_variant_id")
    _reject_construct_number_as_composite_identity(
        subject_id=subject_id,
        study_variant_id=study_variant_id,
        rt_part=rt_part,
        aliases=aliases,
    )
    binding = SubjectBinding(
        subject_id=subject_id,
        study_variant_id=study_variant_id,
        payload_program_id=_text(payload["payload_program_id"], label=f"{label}.payload_program_id"),
        rt_part=rt_part,
        lnrna_part=lnrna_part,
        msd_structure=msd_structure,
        aliases=aliases,
        construct_projection_status=projection_status,
    )
    return binding, lnrna_sequence, rt_cds_sequence


def _reject_construct_number_as_composite_identity(
    *,
    subject_id: str,
    study_variant_id: str,
    rt_part: PartAuthorityRef,
    aliases: Sequence[ReaderAlias],
) -> None:
    if rt_part.owner_study_id == _STUDY_ID:
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


def _project_source_sets(
    *, root: Path, sources: _SourceCache, source_sets: Sequence[object]
) -> tuple[tuple[Mapping[str, object], ...], set[str]]:
    if len(source_sets) != 1:
        raise SubjectBindingContractError("source_sets must contain exactly one catalog projection")
    payload = _mapping(source_sets[0], label="source_sets[0]")
    _require_exact_fields(payload, _SOURCE_SET_FIELDS, label="source_sets[0]")
    source_set_id = _text(payload["source_set_id"], label="source_sets[0].source_set_id")
    if source_set_id != _SOURCE_SET_ID:
        raise SubjectBindingContractError(f"source_sets[0].source_set_id must be {_SOURCE_SET_ID}")
    projection_kind = _text(payload["projection_kind"], label="source_sets[0].projection_kind")
    if projection_kind != _SOURCE_SET_PROJECTION_KIND:
        raise SubjectBindingContractError(f"source_sets[0].projection_kind must be {_SOURCE_SET_PROJECTION_KIND}")
    owner_study_id = _text(payload["owner_study_id"], label="source_sets[0].owner_study_id")
    if owner_study_id != _STUDY_ID:
        raise SubjectBindingContractError(f"source_sets[0].owner_study_id must be {_STUDY_ID}")
    source_ref = _text(payload["source_path"], label="source_sets[0].source_path")
    source_path = _source_path(root, source_ref, label="source_sets[0]")
    catalog = _mapping(sources.load_yaml(source_path), label="source_sets[0].source")
    if catalog.get("catalog_id") != _GENBANK_CATALOG_ID or catalog.get("study_id") != _STUDY_ID:
        raise SubjectBindingContractError("source_sets[0]: unsupported GenBank catalog identity")
    if catalog.get("ok") is not True or _list(catalog.get("errors"), label="source_sets[0].source.errors"):
        raise SubjectBindingContractError("source_sets[0]: GenBank catalog is not valid")
    records = _mapping(catalog.get("records"), label="source_sets[0].source.records")
    msd_manifest_ref = _text(
        payload["msd_source_manifest_path"],
        label="source_sets[0].msd_source_manifest_path",
    )
    msd_manifest_path, msd_records = _load_source_set_msd_records(
        root=root,
        sources=sources,
        source_ref=msd_manifest_ref,
    )
    exclusions = tuple(
        _text(value, label="source_sets[0].exclude_record_ids[]")
        for value in _list(payload["exclude_record_ids"], label="source_sets[0].exclude_record_ids")
    )
    if len(set(exclusions)) != len(exclusions):
        raise SubjectBindingContractError("source_sets[0].exclude_record_ids must be unique")
    missing_exclusions = sorted(set(exclusions) - set(records))
    if missing_exclusions:
        raise SubjectBindingContractError(
            "source_sets[0].exclude_record_ids are absent from the catalog: " + ", ".join(missing_exclusions)
        )
    overrides_payload = _mapping(
        payload["payload_program_overrides"],
        label="source_sets[0].payload_program_overrides",
    )
    overrides = {
        _text(key, label="source_sets[0].payload_program_overrides key"): _text(
            value,
            label=f"source_sets[0].payload_program_overrides.{key}",
        )
        for key, value in overrides_payload.items()
    }
    invalid_overrides = sorted(set(overrides) - (set(records) - set(exclusions)))
    if invalid_overrides:
        raise SubjectBindingContractError(
            "source_sets[0].payload_program_overrides do not name included catalog records: "
            + ", ".join(invalid_overrides)
        )
    default_payload_program_id = _text(
        payload["default_payload_program_id"],
        label="source_sets[0].default_payload_program_id",
    )
    projected: list[Mapping[str, object]] = []
    projection_identities: list[dict[str, str]] = []
    for record_id, raw_record in records.items():
        variant_id = _text(record_id, label="source_sets[0].source.records key")
        if variant_id in exclusions:
            continue
        record = _mapping(raw_record, label=f"source_sets[0].source.records.{variant_id}")
        if _text(record.get("variant_id"), label=f"{variant_id}.variant_id") != variant_id:
            raise SubjectBindingContractError(f"{variant_id}: catalog record key does not match variant_id")
        lnrna = _mapping(record.get("lnrna"), label=f"{variant_id}.lnrna")
        rt_cds = _mapping(record.get("rt_cds"), label=f"{variant_id}.rt_cds")
        lnrna_sequence = _extract_catalog_sequence(
            root=root,
            sources=sources,
            catalog=catalog,
            authority=lnrna,
            label=f"source_sets[0].source.records.{variant_id}.lnrna",
        )
        payload_program_id = overrides.get(variant_id, default_payload_program_id)
        subject_id = _text(record.get("construct_subject_id"), label=f"{variant_id}.construct_subject_id")
        reader_design_id = _text(record.get("reader_design_id"), label=f"{variant_id}.reader_design_id")
        rt_sequence_sha256 = _digest(
            rt_cds.get("sequence_sha256"),
            label=f"{variant_id}.rt_cds.sequence_sha256",
        )
        lnrna_sequence_sha256 = _digest(
            lnrna.get("sequence_sha256"),
            label=f"{variant_id}.lnrna.sequence_sha256",
        )
        projection_status = _text(
            record.get("construct_projection_status"),
            label=f"{variant_id}.construct_projection_status",
        )
        source_path = _text(record.get("source_path"), label=f"{variant_id}.source_path")
        source_sha256 = _digest(record.get("source_sha256"), label=f"{variant_id}.source_sha256")
        resolved_source_path = _source_path(root, source_path, label=f"{variant_id}.source")
        observed_source_sha256 = sources.load_file_sha256(resolved_source_path)
        if observed_source_sha256 != source_sha256:
            raise SubjectBindingContractError(
                f"{variant_id}: GenBank source file digest mismatch: "
                f"declared {source_sha256}, observed {observed_source_sha256}"
            )
        source_kind = _text(record.get("source_kind"), label=f"{variant_id}.source_kind")
        benchling_url = _text(record.get("benchling_url"), label=f"{variant_id}.benchling_url")
        projection_identities.append(
            {
                "record_id": variant_id,
                "study_variant_id": variant_id,
                "subject_id": subject_id,
                "reader_design_id": reader_design_id,
                "source_path": source_path,
                "source_sha256": source_sha256,
                "source_kind": source_kind,
                "benchling_url": benchling_url,
                "rt_sequence_sha256": rt_sequence_sha256,
                "lnrna_sequence_sha256": lnrna_sequence_sha256,
                "payload_program_id": payload_program_id,
                "construct_projection_status": projection_status,
            }
        )
        projected.append(
            {
                "subject_id": subject_id,
                "study_variant_id": variant_id,
                "payload_program_id": payload_program_id,
                "rt_part": _projected_part_ref(
                    owner_study_id=owner_study_id,
                    source_path=source_ref,
                    record_id=variant_id,
                    part_id=_text(rt_cds.get("sequence_id"), label=f"{variant_id}.rt_cds.sequence_id"),
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
                "aliases": [
                    {
                        "namespace": "reader.design_id",
                        "value": reader_design_id,
                    }
                ],
                "construct_projection_status": projection_status,
            }
        )
    declared_projection_digest = _digest(
        payload["projection_sha256"],
        label="source_sets[0].projection_sha256",
    )
    observed_projection_digest = _projection_sha256(projection_identities)
    if observed_projection_digest != declared_projection_digest:
        raise SubjectBindingContractError(
            "source_sets[0]: projected identity digest drifted; "
            f"declared {declared_projection_digest}, observed {observed_projection_digest}"
        )
    return tuple(projected), set(exclusions)


def _projection_sha256(rows: Sequence[Mapping[str, str]]) -> str:
    canonical_rows = sorted(rows, key=lambda row: (row["study_variant_id"], row["subject_id"]))
    canonical_json = json.dumps(canonical_rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _sha256(canonical_json)


def _load_source_set_msd_records(
    *, root: Path, sources: _SourceCache, source_ref: str
) -> tuple[Path, dict[str, Mapping[str, object]]]:
    manifest_path = _source_path(root, source_ref, label="source_sets[0].msd_source_manifest_path")
    manifest = _mapping(sources.load_yaml(manifest_path), label="source_sets[0].msd_source")
    if manifest.get("contract") != "retron_msd_region_record_bundle_v1":
        raise SubjectBindingContractError("source_sets[0]: unsupported MSD source contract")
    indexed: dict[str, Mapping[str, object]] = {}
    for index, raw_record in enumerate(_list(manifest.get("records"), label="source_sets[0].msd_source.records")):
        record = _mapping(raw_record, label=f"source_sets[0].msd_source.records[{index}]")
        variant_id = _text(record.get("variant_id"), label=f"source_sets[0].msd_source.records[{index}].variant_id")
        if variant_id in indexed:
            raise SubjectBindingContractError(f"source_sets[0]: duplicate MSD structure record {variant_id!r}")
        indexed[variant_id] = record
    return manifest_path, indexed


def _projected_msd_structure(
    *,
    sources: _SourceCache,
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
    record_path = _contained_file(
        manifest_path.parent,
        _text(manifest_record.get("record"), label=f"{variant_id}.msd.record"),
        label=f"{variant_id}.msd.record",
    )
    record = _mapping(sources.load_yaml(record_path), label=f"{variant_id}.msd.record")
    if record.get("variant_id") != variant_id:
        raise SubjectBindingContractError(f"{variant_id}: MSD record identity mismatch")
    if require_hairpin_source_closure:
        hairpin_source_digest = _digest(
            record.get("source_sequence_sha256"),
            label=f"{variant_id}.source_sequence_sha256",
        )
        if hairpin_source_digest != lnrna_sequence_sha256:
            raise SubjectBindingContractError(
                f"{variant_id}: hairpin source sequence digest disagrees with catalog lnRNA digest"
            )
    msd_sequence = _text(record.get("msd_sequence_5to3"), label=f"{variant_id}.msd_sequence_5to3").upper()
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
        "sequence_sha256": _digest(
            record.get("msd_sequence_sha256"),
            label=f"{variant_id}.msd_sequence_sha256",
        ),
        "orientation_in_lnrna": orientation,
        "lnrna_span_0": [start, end],
        "structure_materialization_id": manifest_path.parent.name,
        "structure_subject_id": _text(record.get("display_id"), label=f"{variant_id}.display_id"),
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
    *,
    owner_study_id: str,
    source_path: str,
    record_id: str,
    part_id: str,
    sequence_sha256: str,
) -> dict[str, str]:
    return {
        "owner_study_id": owner_study_id,
        "part_id": part_id,
        "authority_kind": _SOURCE_SET_PROJECTION_KIND,
        "source_path": source_path,
        "record_id": record_id,
        "sequence_sha256": sequence_sha256,
    }


def _resolve_part(
    *,
    root: Path,
    sources: _SourceCache,
    payload: Mapping[str, object],
    label: str,
    component: str,
    require_sequence_bytes: bool,
) -> tuple[PartAuthorityRef, str | None]:
    _require_exact_fields(payload, _PART_FIELDS, label=label)
    ref = PartAuthorityRef(
        owner_study_id=_text(payload["owner_study_id"], label=f"{label}.owner_study_id"),
        part_id=_text(payload["part_id"], label=f"{label}.part_id"),
        authority_kind=_text(payload["authority_kind"], label=f"{label}.authority_kind"),
        source_path=_text(payload["source_path"], label=f"{label}.source_path"),
        record_id=_text(payload["record_id"], label=f"{label}.record_id"),
        sequence_sha256=_digest(payload["sequence_sha256"], label=f"{label}.sequence_sha256"),
    )
    source_path = _source_path(root, ref.source_path, label=label)
    if ref.authority_kind == "rt_lnrna_variant_genbank_catalog":
        if ref.owner_study_id != _STUDY_ID:
            raise SubjectBindingContractError(
                f"{label}: owner_study_id must be {_STUDY_ID!r} for the study-owned GenBank catalog"
            )
        source = _mapping(sources.load_yaml(source_path), label=f"{label}.source")
        records = _mapping(source.get("records"), label=f"{label}.source.records")
        record = _mapping(records.get(ref.record_id), label=f"{label}.source.records.{ref.record_id}")
        authority = _mapping(record.get(component), label=f"{label}.{component}")
        observed_id = _text(authority.get("sequence_id"), label=f"{label}.{component}.sequence_id")
        observed_digest = _digest(authority.get("sequence_sha256"), label=f"{label}.{component}.sequence_sha256")
        if component == "lnrna":
            # lnRNA part ids are semantic variant ids; the exact source sequence id remains in the catalog.
            if ref.part_id != ref.record_id:
                raise SubjectBindingContractError(f"{label}: lnRNA part_id must equal catalog record_id")
        elif observed_id != ref.part_id:
            raise SubjectBindingContractError(f"{label}: RT part_id does not match the catalog sequence_id")
        sequence = _extract_catalog_sequence(
            root=root,
            sources=sources,
            catalog=source,
            authority=authority,
            label=label,
        )
        if _sha256(sequence) != observed_digest:
            raise SubjectBindingContractError(f"{label}: source GenBank sequence digest disagrees with catalog")
    elif ref.authority_kind == _RT_PART_PUBLICATION_KIND and component == "rt_cds":
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
        observed_digest = part.cds_sha256
        ref = replace(
            ref,
            provider_ref=part.provider_ref,
            cds_length_nt=part.cds_length_nt,
            terminal_stop_codon=part.terminal_stop_codon,
            protein_sha256=part.protein_sha256,
            protein_length_aa=part.protein_length_aa,
        )
        sequence = None
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
            f"provider reference {part.provider_ref!r} must be resolved through a provider-owned byte authority"
        )
    return ref, sequence


def _resolve_msd_structure(
    *, root: Path, sources: _SourceCache, payload: Mapping[str, object], label: str, lnrna_sequence: str
) -> MsdStructureRef:
    _require_exact_fields(payload, _MSD_FIELDS, label=label)
    span = _span(payload["lnrna_span_0"], label=f"{label}.lnrna_span_0")
    ref = MsdStructureRef(
        owner_study_id=_text(payload["owner_study_id"], label=f"{label}.owner_study_id"),
        source_manifest_path=_text(payload["source_manifest_path"], label=f"{label}.source_manifest_path"),
        variant_id=_text(payload["variant_id"], label=f"{label}.variant_id"),
        sequence_sha256=_digest(payload["sequence_sha256"], label=f"{label}.sequence_sha256"),
        orientation_in_lnrna=_text(payload["orientation_in_lnrna"], label=f"{label}.orientation_in_lnrna"),
        lnrna_span_0=span,
        structure_materialization_id=_text(
            payload["structure_materialization_id"], label=f"{label}.structure_materialization_id"
        ),
        structure_subject_id=_text(payload["structure_subject_id"], label=f"{label}.structure_subject_id"),
    )
    if ref.owner_study_id != "retron_hairpin_design":
        raise SubjectBindingContractError(f"{label}.owner_study_id must be retron_hairpin_design")
    if ref.orientation_in_lnrna not in {"forward", "reverse_complement"}:
        raise SubjectBindingContractError(f"{label}.orientation_in_lnrna is unsupported")
    manifest_path = _source_path(root, ref.source_manifest_path, label=label)
    if ref.structure_materialization_id != manifest_path.parent.name:
        raise SubjectBindingContractError(
            f"{label}: structure_materialization_id does not match the source bundle directory"
        )
    manifest = _mapping(sources.load_yaml(manifest_path), label=f"{label}.source")
    if manifest.get("contract") != "retron_msd_region_record_bundle_v1":
        raise SubjectBindingContractError(f"{label}: unsupported MSD source contract")
    rows = _list(manifest.get("records"), label=f"{label}.source.records")
    matches = [row for row in rows if isinstance(row, Mapping) and row.get("variant_id") == ref.variant_id]
    if len(matches) != 1:
        raise SubjectBindingContractError(f"{label}: expected one MSD record for {ref.variant_id!r}")
    manifest_row = _mapping(matches[0], label=f"{label}.source.records.{ref.variant_id}")
    record_path = _contained_file(
        manifest_path.parent,
        _text(manifest_row.get("record"), label=f"{label}.record"),
        label=f"{label}.record",
    )
    record = _mapping(sources.load_yaml(record_path), label=f"{label}.record")
    if record.get("contract") != "retron_msd_region_record_v1" or record.get("variant_id") != ref.variant_id:
        raise SubjectBindingContractError(f"{label}: MSD record contract or identity mismatch")
    msd_sequence = _text(record.get("msd_sequence_5to3"), label=f"{label}.msd_sequence_5to3").upper()
    observed_digest = _digest(record.get("msd_sequence_sha256"), label=f"{label}.msd_sequence_sha256")
    if _sha256(msd_sequence) != observed_digest or observed_digest != ref.sequence_sha256:
        raise SubjectBindingContractError(f"{label}: MSD source digest mismatch")
    expected = msd_sequence if ref.orientation_in_lnrna == "forward" else str(Seq(msd_sequence).reverse_complement())
    start, end = span
    if end > len(lnrna_sequence) or lnrna_sequence[start:end] != expected:
        raise SubjectBindingContractError(f"{label}: MSD sequence does not match lnRNA span in declared orientation")
    if ref.structure_subject_id != _text(record.get("display_id"), label=f"{label}.display_id"):
        raise SubjectBindingContractError(f"{label}: structure_subject_id does not match MSD display_id")
    return ref


def _extract_catalog_sequence(
    *,
    root: Path,
    sources: _SourceCache,
    catalog: Mapping[str, object],
    authority: Mapping[str, object],
    label: str,
) -> str:
    sequence_id = _text(authority.get("sequence_id"), label=f"{label}.sequence_id")
    source_ref, separator, _record_ref = sequence_id.partition("#")
    if not separator or not source_ref.startswith("genbank:"):
        raise SubjectBindingContractError(f"{label}.sequence_id must use genbank:<file>#<record>")
    source_file = source_ref.removeprefix("genbank:")
    if not source_file or Path(source_file).name != source_file:
        raise SubjectBindingContractError(f"{label}.sequence_id must name one GenBank file")
    genbank_dir_ref = _text(catalog.get("genbank_dir"), label=f"{label}.catalog.genbank_dir")
    genbank_dir = Path(genbank_dir_ref)
    if genbank_dir.is_absolute() or ".." in genbank_dir.parts:
        raise SubjectBindingContractError(f"{label}.catalog.genbank_dir must be repo-relative without parent traversal")
    source_path = _contained_file(
        root,
        (genbank_dir / source_file).as_posix(),
        label=f"{label}.genbank_component",
    )
    genbank_sequence = sources.load_genbank_sequence(source_path)
    start, end = _span(authority.get("span_0"), label=f"{label}.span_0")
    if end > len(genbank_sequence):
        raise SubjectBindingContractError(f"{label}: source span exceeds GenBank record")
    return genbank_sequence[start:end]


def _parse_alias(payload: Mapping[str, object], *, label: str) -> ReaderAlias:
    _require_exact_fields(payload, _ALIAS_FIELDS, label=label)
    namespace = _text(payload["namespace"], label=f"{label}.namespace")
    if namespace not in _ALIAS_NAMESPACES:
        raise SubjectBindingContractError(f"{label}.namespace must be one of {sorted(_ALIAS_NAMESPACES)}")
    return ReaderAlias(namespace=namespace, value=_text(payload["value"], label=f"{label}.value"))


def _require_exact_fields(payload: Mapping[str, object], expected: set[str], *, label: str) -> None:
    observed = set(payload)
    unknown = sorted(observed - expected)
    missing = sorted(expected - observed)
    if unknown:
        raise SubjectBindingContractError(f"{label} has unknown field(s): {', '.join(unknown)}")
    if missing:
        raise SubjectBindingContractError(f"{label} is missing required field(s): {', '.join(missing)}")


def _source_path(root: Path, value: str, *, label: str) -> Path:
    return _contained_file(root, value, label=f"{label}.source_path")


def _contained_file(base: Path, value: str, *, label: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise SubjectBindingContractError(f"{label}: path must be relative without parent traversal")
    resolved_base = Path(base).resolve()
    path = (resolved_base / relative).resolve()
    try:
        path.relative_to(resolved_base)
    except ValueError as exc:
        raise SubjectBindingContractError(f"{label}: resolved path must remain inside its owning directory") from exc
    if not path.is_file():
        raise SubjectBindingContractError(f"{label}: source file is missing: {value}")
    return path


def _digest(value: object, *, label: str) -> str:
    text = _text(value, label=label)
    normalized = text if text.startswith("sha256:") else f"sha256:{text}"
    if len(normalized) != 71 or any(char not in "0123456789abcdef" for char in normalized[7:]):
        raise SubjectBindingContractError(f"{label} must be a lowercase sha256 digest")
    return normalized


def _sha256(sequence: str) -> str:
    return f"sha256:{hashlib.sha256(sequence.encode('utf-8')).hexdigest()}"


def _span(value: object, *, label: str) -> tuple[int, int]:
    rows = _list(value, label=label)
    if len(rows) != 2 or not all(isinstance(item, int) and not isinstance(item, bool) for item in rows):
        raise SubjectBindingContractError(f"{label} must be [start, end] integers")
    start, end = int(rows[0]), int(rows[1])
    if start < 0 or end <= start:
        raise SubjectBindingContractError(f"{label} must be a non-empty zero-based half-open span")
    return start, end


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SubjectBindingContractError(f"{label} must be a non-empty string")
    return value.strip()


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SubjectBindingContractError(f"{label} must be a mapping")
    return value


def _list(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise SubjectBindingContractError(f"{label} must be a list")
    return value


def _load_yaml(path: Path) -> object:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SubjectBindingContractError(f"unable to read {path}: {exc}") from exc


def _resolve_repo_root(repo_root: Path | None) -> Path:
    if repo_root is not None:
        return Path(repo_root).expanduser().resolve()
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


__all__ = [
    "load_registered_subject_binding_materialization",
    "load_registered_subject_bindings",
    "load_resolved_registered_subject_bindings",
    "load_resolved_subject_bindings",
    "load_subject_bindings",
]
