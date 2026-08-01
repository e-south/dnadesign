"""Build study subject bindings from one source-closed Reader record."""

from __future__ import annotations

import hashlib
from io import BytesIO

import pandas as pd

from dnadesign.studies.core.reader_records import ReaderDataframeRecordRef

from ...subject_bindings import SubjectBindingRegistry
from .contracts import (
    READER_EVIDENCE_BINDING_SCHEMA_ID,
    BiologicalReplicateIdentityScope,
    ReaderEvidenceBinding,
    ReaderEvidenceBindingError,
    ReaderEvidenceBindingSet,
)
from .validation import sha256_digest

_DESIGN_NAMESPACE = "reader.design_id"
_ASSAY_SUBJECT_NAMESPACE = "reader.assay_subject_id"


def build_reader_evidence_bindings(
    *,
    record: ReaderDataframeRecordRef,
    subject_registry: SubjectBindingRegistry,
) -> ReaderEvidenceBindingSet:
    """Build exact subject bindings from a verified sample-only Reader dataframe."""

    _validate_sources(record=record, subject_registry=subject_registry)
    try:
        artifact_bytes = record.path.read_bytes()
    except OSError as exc:
        raise ReaderEvidenceBindingError(f"cannot read verified Reader artifact {record.path}: {exc}") from exc
    observed_digest = "sha256:" + hashlib.sha256(artifact_bytes).hexdigest()
    if observed_digest != record.content_digest:
        raise ReaderEvidenceBindingError(
            f"{record.ref}: content digest changed after Reader record resolution; "
            f"expected {record.content_digest}, observed {observed_digest}"
        )
    try:
        frame = pd.read_parquet(BytesIO(artifact_bytes))
    except Exception as exc:
        raise ReaderEvidenceBindingError(f"cannot read verified Reader dataframe {record.path}: {exc}") from exc
    _validate_frame(frame=frame, record=record)

    observation_identity_field = "position"
    grouped: dict[tuple[str | None, str | None], tuple[set[str], set[tuple[str, str]]]] = {}
    for row_index, row in frame.iterrows():
        design_id = _optional_cell_text(row.get("design_id"), label=f"row {row_index}.design_id")
        assay_subject_id = _optional_cell_text(row.get("assay_subject_id"), label=f"row {row_index}.assay_subject_id")
        if design_id is None and assay_subject_id is None:
            raise ReaderEvidenceBindingError(f"row {row_index} has no Reader subject identity")
        observation_identity = _optional_cell_text(
            row.get(observation_identity_field), label=f"row {row_index}.{observation_identity_field}"
        )
        if observation_identity is None:
            raise ReaderEvidenceBindingError(f"row {row_index}.{observation_identity_field} must be populated")
        observation_identities, replicate_identities = grouped.setdefault((design_id, assay_subject_id), (set(), set()))
        observation_identities.add(observation_identity)
        if record.replicate_identity_field is not None:
            condition_value = _optional_cell_text(row.get("treatment"), label=f"row {row_index}.treatment")
            if condition_value is None:
                raise ReaderEvidenceBindingError(
                    "declared biological-replicate identity requires a populated treatment condition"
                )
            replicate_identity = _optional_cell_text(
                row.get(record.replicate_identity_field),
                label=f"row {row_index}.{record.replicate_identity_field}",
            )
            if replicate_identity is None:
                raise ReaderEvidenceBindingError(f"row {row_index}.{record.replicate_identity_field} must be populated")
            replicate_identities.add((condition_value, replicate_identity))

    bindings = tuple(
        _binding_row(
            record=record,
            subject_registry=subject_registry,
            observation_identity_field=observation_identity_field,
            design_id=identity[0],
            assay_subject_id=identity[1],
            observation_identities=tuple(sorted(values[0])),
            biological_replicate_identity_scopes=tuple(
                BiologicalReplicateIdentityScope(condition_value=condition, biological_replicate_id=replicate_id)
                for condition, replicate_id in sorted(values[1])
            ),
        )
        for identity, values in sorted(grouped.items(), key=lambda item: ((item[0][0] or ""), (item[0][1] or "")))
    )
    return ReaderEvidenceBindingSet._from_source_closed_record(
        schema_id=READER_EVIDENCE_BINDING_SCHEMA_ID,
        subject_binding_set_id=subject_registry.binding_set_id,
        rows=bindings,
    )


def _validate_sources(*, record: ReaderDataframeRecordRef, subject_registry: SubjectBindingRegistry) -> None:
    if record.record_schema_version != 6:
        raise ReaderEvidenceBindingError("Reader evidence bindings require record schema v6")
    if record.record_id != "sample_measurements/df":
        raise ReaderEvidenceBindingError("Reader evidence bindings require record 'sample_measurements/df'")
    if type(record.revision) is not int or record.revision < 1:
        raise ReaderEvidenceBindingError("Reader record revision must be a positive integer")
    sha256_digest(record.revision_digest, label="Reader record revision_digest")
    sha256_digest(record.content_digest, label="Reader record content_digest")
    if record.contract_id != "plate_reader.annotated.v1":
        raise ReaderEvidenceBindingError("Reader evidence bindings require contract 'plate_reader.annotated.v1'")
    if not record.is_source_closed:
        raise ReaderEvidenceBindingError(
            "Reader evidence bindings require a source-closed Reader record returned by the public resolver"
        )
    if not isinstance(subject_registry, SubjectBindingRegistry) or not subject_registry.is_source_closed:
        raise ReaderEvidenceBindingError(
            "Reader evidence bindings require a source-closed registry returned by the subject-binding loader"
        )
    if record.replicate_kind not in {"unknown", "biological"}:
        raise ReaderEvidenceBindingError(
            "RT-lnRNA Reader bindings accept biological or unknown replicate declarations; "
            "they never coerce observations to technical replicates"
        )
    if record.replicate_kind == "unknown" and record.replicate_identity_field is not None:
        raise ReaderEvidenceBindingError(
            "unknown replicate identity cannot declare a biological-replicate identity field"
        )


def _validate_frame(*, frame: pd.DataFrame, record: ReaderDataframeRecordRef) -> None:
    if not any(column in frame.columns for column in ("design_id", "assay_subject_id")):
        raise ReaderEvidenceBindingError("Reader dataframe requires design_id and/or assay_subject_id")
    if "position" not in frame.columns:
        raise ReaderEvidenceBindingError("Reader dataframe is missing observation identity field 'position'")
    if record.replicate_identity_field is not None and record.replicate_identity_field not in frame.columns:
        raise ReaderEvidenceBindingError(
            "Reader dataframe is missing declared biological-replicate identity field "
            f"{record.replicate_identity_field!r}"
        )


def _binding_row(
    *,
    record: ReaderDataframeRecordRef,
    subject_registry: SubjectBindingRegistry,
    observation_identity_field: str,
    design_id: str | None,
    assay_subject_id: str | None,
    observation_identities: tuple[str, ...],
    biological_replicate_identity_scopes: tuple[BiologicalReplicateIdentityScope, ...],
) -> ReaderEvidenceBinding:
    resolved: dict[str, str] = {}
    populated_aliases: list[str] = []
    for namespace, value in ((_DESIGN_NAMESPACE, design_id), (_ASSAY_SUBJECT_NAMESPACE, assay_subject_id)):
        if value is None:
            continue
        alias = f"{namespace}:{value}"
        populated_aliases.append(alias)
        subject = subject_registry.subjects_by_alias.get((namespace, value))
        if subject is not None:
            resolved[alias] = subject.subject_id
    subject_ids = set(resolved.values())
    if len(subject_ids) > 1:
        details = ", ".join(f"{alias} -> {subject_id}" for alias, subject_id in sorted(resolved.items()))
        raise ReaderEvidenceBindingError(
            f"{record.experiment_id}: conflicting exact aliases for one Reader row: {details}"
        )
    if set(populated_aliases) - set(resolved):
        subject_id = None
        binding_state = "unbound"
        binding_reason = "partial_exact_subject_alias_match" if resolved else "no_exact_subject_alias_match"
    else:
        subject_id = next(iter(subject_ids), None)
        binding_state = "bound"
        binding_reason = "exact_subject_alias_match"
    return ReaderEvidenceBinding(
        reader_experiment_id=record.experiment_id,
        reader_protocol_id=record.protocol_id,
        reader_replicate_kind=record.replicate_kind,
        reader_replicate_identity_field=record.replicate_identity_field,
        reader_record_id=record.record_id,
        reader_record_kind=record.record_kind,
        reader_record_schema_version=record.record_schema_version,
        reader_record_revision=record.revision,
        reader_record_revision_digest=record.revision_digest,
        reader_record_contract_id=record.contract_id,
        reader_record_content_digest=record.content_digest,
        reader_record_path=record.reader_path,
        raw_design_id=design_id,
        raw_assay_subject_id=assay_subject_id,
        subject_id=subject_id,
        observation_identity_field=observation_identity_field,
        observation_identity_values=observation_identities,
        biological_replicate_identity_scopes=biological_replicate_identity_scopes,
        binding_state=binding_state,
        binding_reason=binding_reason,
    )


def _optional_cell_text(value: object, *, label: str) -> str | None:
    if value is None or pd.isna(value):
        return None
    if not isinstance(value, str) or not value.strip():
        raise ReaderEvidenceBindingError(f"{label} must be a non-empty string or null")
    return value.strip()


__all__: list[str] = []
