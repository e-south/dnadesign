"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reader_evidence/bindings.py

Bind verified Reader identities to exact compositional study subjects.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path

import pandas as pd

from dnadesign.studies.core.reader_records import ReaderDataframeRecordRef

from ..subject_bindings import SubjectBindingRegistry

READER_EVIDENCE_BINDING_SCHEMA_ID = "rt_lnrna_reader_evidence_bindings_v4"
_DESIGN_NAMESPACE = "reader.design_id"
_ASSAY_SUBJECT_NAMESPACE = "reader.assay_subject_id"
_BINDING_SET_SOURCE_CLOSURE_TOKEN = object()
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


class ReaderEvidenceBindingError(ValueError):
    """Raised when Reader identities cannot be bound without ambiguity."""


@dataclass(frozen=True, slots=True)
class BiologicalReplicateIdentityScope:
    """One explicitly declared replicate identity within one source condition."""

    condition_value: str
    biological_replicate_id: str


@dataclass(frozen=True, slots=True)
class ReaderEvidenceBinding:
    """One distinct Reader identity pair and its compositional subject join."""

    reader_experiment_id: str
    reader_protocol_id: str
    reader_replicate_kind: str
    reader_replicate_identity_field: str | None
    reader_record_id: str
    reader_record_kind: str
    reader_record_schema_version: int
    reader_record_revision: int
    reader_record_revision_digest: str
    reader_record_contract_id: str
    reader_record_content_digest: str
    reader_record_path: str
    raw_design_id: str | None
    raw_assay_subject_id: str | None
    subject_id: str | None
    observation_identity_field: str
    observation_identity_values: tuple[str, ...]
    binding_state: str
    binding_reason: str
    biological_replicate_identity_scopes: tuple[BiologicalReplicateIdentityScope, ...] = ()


@dataclass(frozen=True, slots=True)
class ReaderEvidenceBindingSet:
    """Bindings derived from one digest-verified Reader dataframe record."""

    schema_id: str
    subject_binding_set_id: str
    rows: tuple[ReaderEvidenceBinding, ...]
    _source_closure: object | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_binding_set(self)

    @classmethod
    def _from_source_closed_record(
        cls,
        *,
        schema_id: str,
        subject_binding_set_id: str,
        rows: tuple[ReaderEvidenceBinding, ...],
    ) -> ReaderEvidenceBindingSet:
        binding_set = cls(
            schema_id=schema_id,
            subject_binding_set_id=subject_binding_set_id,
            rows=rows,
        )
        object.__setattr__(binding_set, "_source_closure", _BINDING_SET_SOURCE_CLOSURE_TOKEN)
        return binding_set

    @property
    def is_source_closed(self) -> bool:
        return self._source_closure is _BINDING_SET_SOURCE_CLOSURE_TOKEN

    @property
    def unbound_count(self) -> int:
        return sum(row.binding_state == "unbound" for row in self.rows)

    @property
    def artifact_id(self) -> str:
        """Stable semantic identity for this exact Reader record binding."""

        first = self.rows[0]
        return f"{self.schema_id}:{first.reader_experiment_id}:{first.reader_record_id}:r{first.reader_record_revision}"

    @property
    def artifact_digest(self) -> str:
        """Canonical content digest, excluding only the digest field itself."""

        payload = _binding_artifact_payload(self, include_digest=False)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()


def build_reader_evidence_bindings(
    *,
    record: ReaderDataframeRecordRef,
    subject_registry: SubjectBindingRegistry,
) -> ReaderEvidenceBindingSet:
    """Build exact subject bindings from a verified sample-only Reader dataframe.

    The artifact digest and v6 exact-revision contract must already have been checked
    by :func:`resolve_digest_verified_dataframe_record`. Unknown identities are
    retained as explicit ``unbound`` rows. No normalization or fuzzy matching
    is performed.
    """

    if record.record_schema_version != 6:
        raise ReaderEvidenceBindingError("Reader evidence bindings require record schema v6")
    if record.record_id != "sample_measurements/df":
        raise ReaderEvidenceBindingError("Reader evidence bindings require record 'sample_measurements/df'")
    if type(record.revision) is not int or record.revision < 1:
        raise ReaderEvidenceBindingError("Reader record revision must be a positive integer")
    _sha256_digest(record.revision_digest, label="Reader record revision_digest")
    _sha256_digest(record.content_digest, label="Reader record content_digest")
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
    if record.replicate_kind not in {"unknown", "biological"}:
        raise ReaderEvidenceBindingError(
            "RT-lnRNA Reader bindings accept biological or unknown replicate declarations; "
            "they never coerce observations to technical replicates"
        )
    if record.replicate_kind == "unknown" and record.replicate_identity_field is not None:
        raise ReaderEvidenceBindingError(
            "unknown replicate identity cannot declare a biological-replicate identity field"
        )
    observation_identity_field = "position"
    try:
        frame = pd.read_parquet(BytesIO(artifact_bytes))
    except Exception as exc:  # pandas normalizes provider-specific parquet errors
        raise ReaderEvidenceBindingError(f"cannot read verified Reader dataframe {record.path}: {exc}") from exc
    identity_columns = [column for column in ("design_id", "assay_subject_id") if column in frame.columns]
    if not identity_columns:
        raise ReaderEvidenceBindingError("Reader dataframe requires design_id and/or assay_subject_id")
    if observation_identity_field not in frame.columns:
        raise ReaderEvidenceBindingError(
            f"Reader dataframe is missing observation identity field {observation_identity_field!r}"
        )
    if record.replicate_identity_field is not None and record.replicate_identity_field not in frame.columns:
        raise ReaderEvidenceBindingError(
            "Reader dataframe is missing declared biological-replicate identity field "
            f"{record.replicate_identity_field!r}"
        )

    grouped: dict[tuple[str | None, str | None], tuple[set[str], set[tuple[str, str]]]] = {}
    for row_index, row in frame.iterrows():
        design_id = _optional_cell_text(row.get("design_id"), label=f"row {row_index}.design_id")
        assay_subject_id = _optional_cell_text(row.get("assay_subject_id"), label=f"row {row_index}.assay_subject_id")
        if design_id is None and assay_subject_id is None:
            raise ReaderEvidenceBindingError(f"row {row_index} has no Reader subject identity")
        observation_identity = _optional_cell_text(
            row.get(observation_identity_field),
            label=f"row {row_index}.{observation_identity_field}",
        )
        if observation_identity is None:
            raise ReaderEvidenceBindingError(f"row {row_index}.{observation_identity_field} must be populated")
        observation_identities, biological_replicate_identities = grouped.setdefault(
            (design_id, assay_subject_id),
            (set(), set()),
        )
        observation_identities.add(observation_identity)
        if record.replicate_identity_field is not None:
            condition_value = _optional_cell_text(row.get("treatment"), label=f"row {row_index}.treatment")
            if condition_value is None:
                raise ReaderEvidenceBindingError(
                    "declared biological-replicate identity requires a populated treatment condition"
                )
            biological_replicate_identity = _optional_cell_text(
                row.get(record.replicate_identity_field),
                label=f"row {row_index}.{record.replicate_identity_field}",
            )
            if biological_replicate_identity is None:
                raise ReaderEvidenceBindingError(f"row {row_index}.{record.replicate_identity_field} must be populated")
            biological_replicate_identities.add((condition_value, biological_replicate_identity))

    bindings = tuple(
        _binding_row(
            record=record,
            subject_registry=subject_registry,
            observation_identity_field=observation_identity_field,
            design_id=identity[0],
            assay_subject_id=identity[1],
            observation_identities=tuple(sorted(identity_values[0])),
            biological_replicate_identity_scopes=tuple(
                BiologicalReplicateIdentityScope(condition_value=condition, biological_replicate_id=replicate_id)
                for condition, replicate_id in sorted(identity_values[1])
            ),
        )
        for identity, identity_values in sorted(
            grouped.items(), key=lambda item: ((item[0][0] or ""), (item[0][1] or ""))
        )
    )
    return ReaderEvidenceBindingSet._from_source_closed_record(
        schema_id=READER_EVIDENCE_BINDING_SCHEMA_ID,
        subject_binding_set_id=subject_registry.binding_set_id,
        rows=bindings,
    )


def materialize_reader_evidence_bindings_json(
    binding_set: ReaderEvidenceBindingSet,
    destination: Path,
) -> Path:
    """Publish a validated, immutable JSON evidence-binding artifact."""

    _validate_binding_set(binding_set)
    if not binding_set.is_source_closed:
        raise ReaderEvidenceBindingError(
            "evidence-binding publication requires a source-closed set returned by the binding builder"
        )
    payload = _binding_artifact_payload(binding_set, include_digest=True)
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
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReaderEvidenceBindingError(f"cannot read evidence-binding artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReaderEvidenceBindingError("evidence-binding artifact must be an object")
    _require_exact_fields(payload, _ARTIFACT_FIELDS, label="evidence-binding artifact")
    bindings_payload = payload["bindings"]
    if not isinstance(bindings_payload, list):
        raise ReaderEvidenceBindingError("evidence-binding artifact.bindings must be an array")
    rows: list[ReaderEvidenceBinding] = []
    for index, value in enumerate(bindings_payload):
        label = f"evidence-binding artifact.bindings[{index}]"
        if not isinstance(value, dict):
            raise ReaderEvidenceBindingError(f"{label} must be an object")
        _require_exact_fields(value, _BINDING_FIELDS, label=label)
        row_payload = dict(value)
        observation_identity_values = row_payload["observation_identity_values"]
        if not isinstance(observation_identity_values, list):
            raise ReaderEvidenceBindingError(f"{label}.observation_identity_values must be an array")
        row_payload["observation_identity_values"] = tuple(observation_identity_values)
        biological_replicate_identity_scopes = row_payload["biological_replicate_identity_scopes"]
        if not isinstance(biological_replicate_identity_scopes, list):
            raise ReaderEvidenceBindingError(f"{label}.biological_replicate_identity_scopes must be an array")
        try:
            row_payload["biological_replicate_identity_scopes"] = tuple(
                BiologicalReplicateIdentityScope(**scope) for scope in biological_replicate_identity_scopes
            )
        except (TypeError, AttributeError) as exc:
            raise ReaderEvidenceBindingError(
                f"{label}.biological_replicate_identity_scopes entries must be objects"
            ) from exc
        try:
            rows.append(ReaderEvidenceBinding(**row_payload))
        except TypeError as exc:
            raise ReaderEvidenceBindingError(f"{label} is malformed") from exc

    declared = ReaderEvidenceBindingSet(
        schema_id=payload["schema_id"],
        subject_binding_set_id=payload["subject_binding_set_id"],
        rows=tuple(rows),
    )
    binding_count = _nonnegative_integer(payload["binding_count"], label="artifact.binding_count")
    unbound_count = _nonnegative_integer(payload["unbound_count"], label="artifact.unbound_count")
    if binding_count != len(declared.rows):
        raise ReaderEvidenceBindingError("evidence-binding artifact.binding_count mismatch")
    if unbound_count != declared.unbound_count:
        raise ReaderEvidenceBindingError("evidence-binding artifact.unbound_count mismatch")
    artifact_id = _required_text(payload["artifact_id"], label="artifact.artifact_id")
    if artifact_id != declared.artifact_id:
        raise ReaderEvidenceBindingError("evidence-binding artifact_id mismatch")
    artifact_digest = _sha256_digest(payload["artifact_digest"], label="artifact.artifact_digest")
    if artifact_digest != declared.artifact_digest:
        raise ReaderEvidenceBindingError("evidence-binding artifact_digest mismatch")
    rederived = build_reader_evidence_bindings(record=record, subject_registry=subject_registry)
    if _binding_artifact_payload(declared, include_digest=True) != _binding_artifact_payload(
        rederived, include_digest=True
    ):
        raise ReaderEvidenceBindingError(
            "evidence-binding artifact no longer matches the current Reader record and subject registry"
        )
    return rederived


def _binding_artifact_payload(
    binding_set: ReaderEvidenceBindingSet,
    *,
    include_digest: bool,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_id": binding_set.schema_id,
        "artifact_id": binding_set.artifact_id,
        "subject_binding_set_id": binding_set.subject_binding_set_id,
        "binding_count": len(binding_set.rows),
        "unbound_count": binding_set.unbound_count,
        "bindings": [asdict(row) for row in binding_set.rows],
    }
    if include_digest:
        payload["artifact_digest"] = binding_set.artifact_digest
    return payload


def _validate_binding_set(binding_set: ReaderEvidenceBindingSet) -> None:
    if not isinstance(binding_set, ReaderEvidenceBindingSet):
        raise ReaderEvidenceBindingError("binding_set must be ReaderEvidenceBindingSet")
    if binding_set.schema_id != READER_EVIDENCE_BINDING_SCHEMA_ID:
        raise ReaderEvidenceBindingError(f"binding_set.schema_id must equal {READER_EVIDENCE_BINDING_SCHEMA_ID!r}")
    _required_text(binding_set.subject_binding_set_id, label="binding_set.subject_binding_set_id")
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
        _required_text(getattr(row, field_name), label=f"{label}.{field_name}")
    if row.reader_replicate_kind not in {"unknown", "biological"}:
        raise ReaderEvidenceBindingError(f"{label}.reader_replicate_kind must be unknown or biological")
    declared_identity = row.reader_replicate_identity_field
    if declared_identity is not None:
        _required_text(declared_identity, label=f"{label}.reader_replicate_identity_field")
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
    _sha256_digest(row.reader_record_revision_digest, label=f"{label}.reader_record_revision_digest")
    if row.reader_record_contract_id != "plate_reader.annotated.v1":
        raise ReaderEvidenceBindingError(f"{label}.reader_record_contract_id must equal 'plate_reader.annotated.v1'")
    _sha256_digest(row.reader_record_content_digest, label=f"{label}.reader_record_content_digest")
    record_path = Path(row.reader_record_path)
    if record_path.is_absolute() or ".." in record_path.parts:
        raise ReaderEvidenceBindingError(f"{label}.reader_record_path must be outputs-relative")
    if row.raw_design_id is None and row.raw_assay_subject_id is None:
        raise ReaderEvidenceBindingError(f"{label} requires at least one raw Reader identity")
    for field_name in ("raw_design_id", "raw_assay_subject_id", "subject_id"):
        value = getattr(row, field_name)
        if value is not None:
            _required_text(value, label=f"{label}.{field_name}")
    if not isinstance(row.observation_identity_values, tuple) or not row.observation_identity_values:
        raise ReaderEvidenceBindingError(f"{label}.observation_identity_values must be a non-empty tuple")
    normalized_observation_identities = tuple(
        _required_text(value, label=f"{label}.observation_identity_values[]")
        for value in row.observation_identity_values
    )
    if len(set(normalized_observation_identities)) != len(normalized_observation_identities):
        raise ReaderEvidenceBindingError(f"{label}.observation_identity_values must be unique")
    if row.reader_replicate_kind == "unknown" and declared_identity is not None:
        raise ReaderEvidenceBindingError(
            f"{label}.reader_replicate_identity_field must be null when reader_replicate_kind is unknown"
        )
    if not isinstance(row.biological_replicate_identity_scopes, tuple):
        raise ReaderEvidenceBindingError(f"{label}.biological_replicate_identity_scopes must be a tuple")
    normalized_replicate_scopes = tuple(
        (
            _required_text(
                scope.condition_value, label=f"{label}.biological_replicate_identity_scopes[].condition_value"
            ),
            _required_text(
                scope.biological_replicate_id,
                label=f"{label}.biological_replicate_identity_scopes[].biological_replicate_id",
            ),
        )
        for scope in row.biological_replicate_identity_scopes
        if isinstance(scope, BiologicalReplicateIdentityScope)
    )
    if len(normalized_replicate_scopes) != len(row.biological_replicate_identity_scopes):
        raise ReaderEvidenceBindingError(
            f"{label}.biological_replicate_identity_scopes entries must be BiologicalReplicateIdentityScope"
        )
    if len(set(normalized_replicate_scopes)) != len(normalized_replicate_scopes):
        raise ReaderEvidenceBindingError(f"{label}.biological_replicate_identity_scopes must be unique")
    if declared_identity is None and normalized_replicate_scopes:
        raise ReaderEvidenceBindingError(
            f"{label}.biological_replicate_identity_scopes must be empty when identity is unknown"
        )
    if declared_identity is not None and not normalized_replicate_scopes:
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
    observed = set(payload)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append("missing=" + ", ".join(missing))
        if unknown:
            details.append("unknown=" + ", ".join(unknown))
        raise ReaderEvidenceBindingError(f"{label} has invalid fields: {'; '.join(details)}")


def _nonnegative_integer(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ReaderEvidenceBindingError(f"{label} must be a non-negative integer")
    return value


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
    if design_id is not None:
        populated_aliases.append(f"{_DESIGN_NAMESPACE}:{design_id}")
        subject = subject_registry.subjects_by_alias.get((_DESIGN_NAMESPACE, design_id))
        if subject is not None:
            resolved[f"{_DESIGN_NAMESPACE}:{design_id}"] = subject.subject_id
    if assay_subject_id is not None:
        populated_aliases.append(f"{_ASSAY_SUBJECT_NAMESPACE}:{assay_subject_id}")
        subject = subject_registry.subjects_by_alias.get((_ASSAY_SUBJECT_NAMESPACE, assay_subject_id))
        if subject is not None:
            resolved[f"{_ASSAY_SUBJECT_NAMESPACE}:{assay_subject_id}"] = subject.subject_id
    subject_ids = set(resolved.values())
    if len(subject_ids) > 1:
        details = ", ".join(f"{alias} -> {subject_id}" for alias, subject_id in sorted(resolved.items()))
        raise ReaderEvidenceBindingError(
            f"{record.experiment_id}: conflicting exact aliases for one Reader row: {details}"
        )
    unresolved_aliases = set(populated_aliases) - set(resolved)
    if unresolved_aliases:
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


def _required_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReaderEvidenceBindingError(f"{label} must be a non-empty string")
    return value.strip()


def _sha256_digest(value: object, *, label: str) -> str:
    token = _required_text(value, label=label)
    if not token.startswith("sha256:") or len(token) != 71:
        raise ReaderEvidenceBindingError(f"{label} must be a sha256 digest")
    if any(character not in "0123456789abcdef" for character in token[7:]):
        raise ReaderEvidenceBindingError(f"{label} must be a lowercase sha256 digest")
    return token


def _optional_cell_text(value: object, *, label: str) -> str | None:
    if value is None or pd.isna(value):
        return None
    if not isinstance(value, str) or not value.strip():
        raise ReaderEvidenceBindingError(f"{label} must be a non-empty string or null")
    return value.strip()


__all__ = [
    "READER_EVIDENCE_BINDING_SCHEMA_ID",
    "ReaderEvidenceBinding",
    "ReaderEvidenceBindingError",
    "ReaderEvidenceBindingSet",
    "build_reader_evidence_bindings",
    "load_reader_evidence_bindings_json",
    "materialize_reader_evidence_bindings_json",
]
