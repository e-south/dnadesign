"""Create-only publication and source-closed loading for binding artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from dnadesign.studies.core.reader_records import ReaderDataframeRecordRef

from ...subject_bindings import SubjectBindingRegistry
from .building import build_reader_evidence_bindings
from .contracts import (
    BiologicalReplicateIdentityScope,
    ReaderEvidenceBinding,
    ReaderEvidenceBindingError,
    ReaderEvidenceBindingSet,
)
from .projection import binding_artifact_payload
from .validation import sha256_digest, validate_binding_set

_ARTIFACT_FIELDS = {
    "schema_id",
    "artifact_id",
    "artifact_digest",
    "subject_binding_set_id",
    "binding_count",
    "unbound_count",
    "bindings",
}
_BINDING_FIELDS = {
    "reader_experiment_id",
    "reader_protocol_id",
    "reader_replicate_kind",
    "reader_replicate_identity_field",
    "reader_record_id",
    "reader_record_kind",
    "reader_record_schema_version",
    "reader_record_revision",
    "reader_record_revision_digest",
    "reader_record_contract_id",
    "reader_record_content_digest",
    "reader_record_path",
    "raw_design_id",
    "raw_assay_subject_id",
    "subject_id",
    "observation_identity_field",
    "observation_identity_values",
    "biological_replicate_identity_scopes",
    "binding_state",
    "binding_reason",
}


def materialize_reader_evidence_bindings_json(
    binding_set: ReaderEvidenceBindingSet,
    destination: Path,
) -> Path:
    """Publish a validated, immutable JSON evidence-binding artifact."""

    validate_binding_set(binding_set)
    if not binding_set.is_source_closed:
        raise ReaderEvidenceBindingError(
            "evidence-binding publication requires a source-closed set returned by the binding builder"
        )
    payload = binding_artifact_payload(binding_set, include_digest=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_create_only_atomic(path, encoded)
    return path


def load_reader_evidence_bindings_json(
    source: Path,
    *,
    record: ReaderDataframeRecordRef,
    subject_registry: SubjectBindingRegistry,
) -> ReaderEvidenceBindingSet:
    """Validate a saved artifact by rederiving it from both current source owners."""

    path = Path(source).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReaderEvidenceBindingError(f"cannot read evidence-binding artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReaderEvidenceBindingError("evidence-binding artifact must be an object")
    _require_exact_fields(payload, _ARTIFACT_FIELDS, label="evidence-binding artifact")
    bindings_payload = payload["bindings"]
    if not isinstance(bindings_payload, list):
        raise ReaderEvidenceBindingError("evidence-binding artifact.bindings must be an array")
    rows = tuple(_binding_from_payload(value, index=index) for index, value in enumerate(bindings_payload))
    declared = ReaderEvidenceBindingSet(
        schema_id=payload["schema_id"], subject_binding_set_id=payload["subject_binding_set_id"], rows=rows
    )
    if _nonnegative_integer(payload["binding_count"], label="artifact.binding_count") != len(declared.rows):
        raise ReaderEvidenceBindingError("evidence-binding artifact.binding_count mismatch")
    if _nonnegative_integer(payload["unbound_count"], label="artifact.unbound_count") != declared.unbound_count:
        raise ReaderEvidenceBindingError("evidence-binding artifact.unbound_count mismatch")
    if _required_text(payload["artifact_id"], label="artifact.artifact_id") != declared.artifact_id:
        raise ReaderEvidenceBindingError("evidence-binding artifact_id mismatch")
    if sha256_digest(payload["artifact_digest"], label="artifact.artifact_digest") != declared.artifact_digest:
        raise ReaderEvidenceBindingError("evidence-binding artifact_digest mismatch")
    rederived = build_reader_evidence_bindings(record=record, subject_registry=subject_registry)
    if binding_artifact_payload(declared, include_digest=True) != binding_artifact_payload(
        rederived, include_digest=True
    ):
        raise ReaderEvidenceBindingError(
            "evidence-binding artifact no longer matches the current Reader record and subject registry"
        )
    return rederived


def _binding_from_payload(value: object, *, index: int) -> ReaderEvidenceBinding:
    label = f"evidence-binding artifact.bindings[{index}]"
    if not isinstance(value, dict):
        raise ReaderEvidenceBindingError(f"{label} must be an object")
    _require_exact_fields(value, _BINDING_FIELDS, label=label)
    row_payload = dict(value)
    observation_values = row_payload["observation_identity_values"]
    if not isinstance(observation_values, list):
        raise ReaderEvidenceBindingError(f"{label}.observation_identity_values must be an array")
    row_payload["observation_identity_values"] = tuple(observation_values)
    replicate_scopes = row_payload["biological_replicate_identity_scopes"]
    if not isinstance(replicate_scopes, list):
        raise ReaderEvidenceBindingError(f"{label}.biological_replicate_identity_scopes must be an array")
    try:
        row_payload["biological_replicate_identity_scopes"] = tuple(
            BiologicalReplicateIdentityScope(**scope) for scope in replicate_scopes
        )
    except (TypeError, AttributeError) as exc:
        raise ReaderEvidenceBindingError(
            f"{label}.biological_replicate_identity_scopes entries must be objects"
        ) from exc
    try:
        return ReaderEvidenceBinding(**row_payload)
    except TypeError as exc:
        raise ReaderEvidenceBindingError(f"{label} is malformed") from exc


def _write_create_only_atomic(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise ReaderEvidenceBindingError(f"evidence-binding artifact already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.chmod(0o644)
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise ReaderEvidenceBindingError(f"evidence-binding artifact already exists: {path}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ReaderEvidenceBindingError(f"duplicate JSON field {key!r}")
        payload[key] = value
    return payload


def _require_exact_fields(payload: dict[str, object], expected: set[str], *, label: str) -> None:
    missing = sorted(expected - set(payload))
    unknown = sorted(set(payload) - expected)
    if missing or unknown:
        details = (["missing=" + ", ".join(missing)] if missing else []) + (
            ["unknown=" + ", ".join(unknown)] if unknown else []
        )
        raise ReaderEvidenceBindingError(f"{label} has invalid fields: {'; '.join(details)}")


def _required_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReaderEvidenceBindingError(f"{label} must be a non-empty string")
    return value.strip()


def _nonnegative_integer(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ReaderEvidenceBindingError(f"{label} must be a non-negative integer")
    return value


__all__: list[str] = []
