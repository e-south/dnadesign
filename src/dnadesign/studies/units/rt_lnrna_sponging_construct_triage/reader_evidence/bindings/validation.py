"""Validation primitives for Reader evidence-binding contracts."""

from __future__ import annotations

from pathlib import Path

from .contracts import (
    READER_EVIDENCE_BINDING_SCHEMA_ID,
    BiologicalReplicateIdentityScope,
    ReaderEvidenceBinding,
    ReaderEvidenceBindingError,
    ReaderEvidenceBindingSet,
)


def required_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReaderEvidenceBindingError(f"{label} must be a non-empty string")
    return value.strip()


def sha256_digest(value: object, *, label: str) -> str:
    token = required_text(value, label=label)
    if not token.startswith("sha256:") or len(token) != 71:
        raise ReaderEvidenceBindingError(f"{label} must be a sha256 digest")
    if any(character not in "0123456789abcdef" for character in token[7:]):
        raise ReaderEvidenceBindingError(f"{label} must be a lowercase sha256 digest")
    return token


def validate_binding_set(binding_set: ReaderEvidenceBindingSet) -> None:
    if not isinstance(binding_set, ReaderEvidenceBindingSet):
        raise ReaderEvidenceBindingError("binding_set must be ReaderEvidenceBindingSet")
    if binding_set.schema_id != READER_EVIDENCE_BINDING_SCHEMA_ID:
        raise ReaderEvidenceBindingError(f"binding_set.schema_id must equal {READER_EVIDENCE_BINDING_SCHEMA_ID!r}")
    required_text(binding_set.subject_binding_set_id, label="binding_set.subject_binding_set_id")
    if not isinstance(binding_set.rows, tuple) or not binding_set.rows:
        raise ReaderEvidenceBindingError("binding_set.rows must be a non-empty tuple")

    record_identities: set[tuple[object, ...]] = set()
    reader_identities: set[tuple[str | None, str | None]] = set()
    for index, row in enumerate(binding_set.rows):
        _validate_binding(row, label=f"binding_set.rows[{index}]")
        record_identities.add(
            (
                row.reader_experiment_id,
                row.reader_protocol_id,
                row.reader_replicate_kind,
                row.reader_replicate_identity_field,
                row.reader_record_id,
                row.reader_record_kind,
                row.reader_record_schema_version,
                row.reader_record_revision,
                row.reader_record_revision_digest,
                row.reader_record_contract_id,
                row.reader_record_content_digest,
                row.reader_record_path,
            )
        )
        reader_identity = (row.raw_design_id, row.raw_assay_subject_id)
        if reader_identity in reader_identities:
            raise ReaderEvidenceBindingError("duplicate Reader identity pair in binding set")
        reader_identities.add(reader_identity)
    if len(record_identities) != 1:
        raise ReaderEvidenceBindingError("binding_set.rows must all cite one exact Reader record identity")


def _validate_binding(row: ReaderEvidenceBinding, *, label: str) -> None:
    if not isinstance(row, ReaderEvidenceBinding):
        raise ReaderEvidenceBindingError(f"{label} must be ReaderEvidenceBinding")
    for field_name in (
        "reader_experiment_id",
        "reader_protocol_id",
        "reader_record_id",
        "reader_record_kind",
        "reader_record_contract_id",
        "reader_record_path",
        "observation_identity_field",
        "binding_state",
        "binding_reason",
    ):
        required_text(getattr(row, field_name), label=f"{label}.{field_name}")
    if row.reader_replicate_kind not in {"unknown", "biological"}:
        raise ReaderEvidenceBindingError(f"{label}.reader_replicate_kind must be unknown or biological")
    declared_identity = row.reader_replicate_identity_field
    if declared_identity is not None:
        required_text(declared_identity, label=f"{label}.reader_replicate_identity_field")
    if row.observation_identity_field != "position":
        raise ReaderEvidenceBindingError(
            f"{label}.observation_identity_field must equal 'position'; observation identity is distinct "
            "from biological-replicate identity"
        )
    if row.reader_record_id != "sample_measurements/df":
        raise ReaderEvidenceBindingError(f"{label}.reader_record_id must equal 'sample_measurements/df'")
    if row.reader_record_kind != "dataframe_artifact":
        raise ReaderEvidenceBindingError(f"{label}.reader_record_kind must equal 'dataframe_artifact'")
    if row.reader_record_schema_version != 6:
        raise ReaderEvidenceBindingError(f"{label}.reader_record_schema_version must equal 6")
    if type(row.reader_record_revision) is not int or row.reader_record_revision < 1:
        raise ReaderEvidenceBindingError(f"{label}.reader_record_revision must be a positive integer")
    sha256_digest(row.reader_record_revision_digest, label=f"{label}.reader_record_revision_digest")
    if row.reader_record_contract_id != "plate_reader.annotated.v1":
        raise ReaderEvidenceBindingError(f"{label}.reader_record_contract_id must equal 'plate_reader.annotated.v1'")
    sha256_digest(row.reader_record_content_digest, label=f"{label}.reader_record_content_digest")
    record_path = Path(row.reader_record_path)
    if record_path.is_absolute() or ".." in record_path.parts:
        raise ReaderEvidenceBindingError(f"{label}.reader_record_path must be outputs-relative")
    if row.raw_design_id is None and row.raw_assay_subject_id is None:
        raise ReaderEvidenceBindingError(f"{label} requires at least one raw Reader identity")
    for field_name in ("raw_design_id", "raw_assay_subject_id", "subject_id"):
        value = getattr(row, field_name)
        if value is not None:
            required_text(value, label=f"{label}.{field_name}")
    if not isinstance(row.observation_identity_values, tuple) or not row.observation_identity_values:
        raise ReaderEvidenceBindingError(f"{label}.observation_identity_values must be a non-empty tuple")
    observation_ids = tuple(
        required_text(value, label=f"{label}.observation_identity_values[]")
        for value in row.observation_identity_values
    )
    if len(set(observation_ids)) != len(observation_ids):
        raise ReaderEvidenceBindingError(f"{label}.observation_identity_values must be unique")
    if row.reader_replicate_kind == "unknown" and declared_identity is not None:
        raise ReaderEvidenceBindingError(
            f"{label}.reader_replicate_identity_field must be null when reader_replicate_kind is unknown"
        )
    if not isinstance(row.biological_replicate_identity_scopes, tuple):
        raise ReaderEvidenceBindingError(f"{label}.biological_replicate_identity_scopes must be a tuple")
    replicate_scopes = tuple(
        (
            required_text(
                scope.condition_value,
                label=f"{label}.biological_replicate_identity_scopes[].condition_value",
            ),
            required_text(
                scope.biological_replicate_id,
                label=f"{label}.biological_replicate_identity_scopes[].biological_replicate_id",
            ),
        )
        for scope in row.biological_replicate_identity_scopes
        if isinstance(scope, BiologicalReplicateIdentityScope)
    )
    if len(replicate_scopes) != len(row.biological_replicate_identity_scopes):
        raise ReaderEvidenceBindingError(
            f"{label}.biological_replicate_identity_scopes entries must be BiologicalReplicateIdentityScope"
        )
    if len(set(replicate_scopes)) != len(replicate_scopes):
        raise ReaderEvidenceBindingError(f"{label}.biological_replicate_identity_scopes must be unique")
    if declared_identity is None and replicate_scopes:
        raise ReaderEvidenceBindingError(
            f"{label}.biological_replicate_identity_scopes must be empty when identity is unknown"
        )
    if declared_identity is not None and not replicate_scopes:
        raise ReaderEvidenceBindingError(
            f"{label}.biological_replicate_identity_scopes must preserve declared scoped identities"
        )
    if row.binding_state == "bound":
        if row.subject_id is None or row.binding_reason != "exact_subject_alias_match":
            raise ReaderEvidenceBindingError(f"{label} has inconsistent bound state")
    elif row.binding_state == "unbound":
        if row.subject_id is not None or row.binding_reason not in {
            "no_exact_subject_alias_match",
            "partial_exact_subject_alias_match",
        }:
            raise ReaderEvidenceBindingError(f"{label} has inconsistent unbound state")
    else:
        raise ReaderEvidenceBindingError(f"{label}.binding_state must be bound or unbound")


__all__: list[str] = []
