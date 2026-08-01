"""Immutable contracts for study-owned Reader evidence bindings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

READER_EVIDENCE_BINDING_SCHEMA_ID = "rt_lnrna_reader_evidence_bindings_v4"
_SOURCE_CLOSURE_TOKEN = object()


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
        from .validation import validate_binding_set

        validate_binding_set(self)

    @classmethod
    def _from_source_closed_record(
        cls,
        *,
        schema_id: str,
        subject_binding_set_id: str,
        rows: tuple[ReaderEvidenceBinding, ...],
    ) -> ReaderEvidenceBindingSet:
        binding_set = cls(schema_id=schema_id, subject_binding_set_id=subject_binding_set_id, rows=rows)
        object.__setattr__(binding_set, "_source_closure", _SOURCE_CLOSURE_TOKEN)
        return binding_set

    @property
    def is_source_closed(self) -> bool:
        return self._source_closure is _SOURCE_CLOSURE_TOKEN

    @property
    def unbound_count(self) -> int:
        return sum(row.binding_state == "unbound" for row in self.rows)

    @property
    def artifact_id(self) -> str:
        first = self.rows[0]
        return f"{self.schema_id}:{first.reader_experiment_id}:{first.reader_record_id}:r{first.reader_record_revision}"

    @property
    def artifact_digest(self) -> str:
        from .projection import binding_artifact_payload

        payload = binding_artifact_payload(self, include_digest=False)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()


__all__ = [
    "BiologicalReplicateIdentityScope",
    "READER_EVIDENCE_BINDING_SCHEMA_ID",
    "ReaderEvidenceBinding",
    "ReaderEvidenceBindingError",
    "ReaderEvidenceBindingSet",
]
