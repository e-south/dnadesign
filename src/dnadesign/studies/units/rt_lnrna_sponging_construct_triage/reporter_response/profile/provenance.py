"""Source-closed Reader evidence provenance for reporter-response profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .._contract_values import ReporterResponseContractError
from .._contract_values import positive_integer as _positive_integer
from .._contract_values import required_text as _required_text
from .._contract_values import sha256_digest as _sha256_digest

if TYPE_CHECKING:
    from ...reader_evidence import ReaderEvidenceBindingSet


@dataclass(frozen=True, slots=True)
class ReaderEvidenceProvenance:
    """Exact Reader record and evidence-binding artifact identities."""

    raw_design_id: str | None
    raw_assay_subject_id: str | None
    reader_experiment_id: str
    reader_protocol_id: str
    reader_record_id: str
    reader_record_kind: str
    reader_record_revision: int
    reader_record_revision_digest: str
    reader_record_content_digest: str
    reader_record_schema_version: int
    reader_record_contract_id: str
    reader_record_path: str
    evidence_binding_artifact_id: str
    evidence_binding_artifact_digest: str
    _bound_subject_id: str | None = field(default=None, init=False, repr=False, compare=False)
    _source_closed: bool = field(default=False, init=False, repr=False, compare=False)
    _declared_biological_replicate_scopes: tuple[tuple[str, str], ...] = field(
        default=(), init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.raw_design_id is None and self.raw_assay_subject_id is None:
            raise ReporterResponseContractError("provenance requires at least one raw Reader identity")
        for name in ("raw_design_id", "raw_assay_subject_id"):
            value = getattr(self, name)
            if value is not None:
                _required_text(value, field_name=name)
        for name in (
            "reader_experiment_id",
            "reader_protocol_id",
            "reader_record_id",
            "reader_record_kind",
            "reader_record_contract_id",
            "reader_record_path",
            "evidence_binding_artifact_id",
        ):
            _required_text(getattr(self, name), field_name=name)
        _positive_integer(self.reader_record_revision, field_name="reader_record_revision")
        if self.reader_record_schema_version != 6:
            raise ReporterResponseContractError("reader_record_schema_version must equal 6")
        if self.reader_record_kind != "dataframe_artifact":
            raise ReporterResponseContractError("reader_record_kind must equal dataframe_artifact")
        for name in (
            "reader_record_revision_digest",
            "reader_record_content_digest",
            "evidence_binding_artifact_digest",
        ):
            _sha256_digest(getattr(self, name), field_name=name)
        record_path = Path(self.reader_record_path)
        if record_path.is_absolute() or ".." in record_path.parts:
            raise ReporterResponseContractError("reader_record_path must be outputs-relative")

    @classmethod
    def _from_source_closed_bindings(
        cls,
        *,
        evidence_bindings: ReaderEvidenceBindingSet,
        subject_id: str,
        raw_design_id: str | None,
        raw_assay_subject_id: str | None,
    ) -> ReaderEvidenceProvenance:
        from ...reader_evidence import ReaderEvidenceBindingSet

        if not isinstance(evidence_bindings, ReaderEvidenceBindingSet) or not evidence_bindings.is_source_closed:
            raise ReporterResponseContractError(
                "reporter-response provenance requires a source-closed Reader evidence-binding set"
            )
        matches = tuple(
            row
            for row in evidence_bindings.rows
            if row.binding_state == "bound"
            and row.subject_id == subject_id
            and row.raw_design_id == raw_design_id
            and row.raw_assay_subject_id == raw_assay_subject_id
        )
        if len(matches) != 1:
            raise ReporterResponseContractError(
                f"subject {subject_id!r} and Reader identity {(raw_design_id, raw_assay_subject_id)!r} "
                "require exactly one bound Reader evidence-binding row; "
                f"observed {len(matches)}"
            )
        row = matches[0]
        provenance = cls(
            raw_design_id=row.raw_design_id,
            raw_assay_subject_id=row.raw_assay_subject_id,
            reader_experiment_id=row.reader_experiment_id,
            reader_protocol_id=row.reader_protocol_id,
            reader_record_id=row.reader_record_id,
            reader_record_kind=row.reader_record_kind,
            reader_record_revision=row.reader_record_revision,
            reader_record_revision_digest=row.reader_record_revision_digest,
            reader_record_content_digest=row.reader_record_content_digest,
            reader_record_schema_version=row.reader_record_schema_version,
            reader_record_contract_id=row.reader_record_contract_id,
            reader_record_path=row.reader_record_path,
            evidence_binding_artifact_id=evidence_bindings.artifact_id,
            evidence_binding_artifact_digest=evidence_bindings.artifact_digest,
        )
        object.__setattr__(provenance, "_bound_subject_id", subject_id)
        object.__setattr__(provenance, "_source_closed", True)
        object.__setattr__(
            provenance,
            "_declared_biological_replicate_scopes",
            tuple(
                sorted(
                    (scope.condition_value, scope.biological_replicate_id)
                    for scope in row.biological_replicate_identity_scopes
                )
            ),
        )
        return provenance

    @property
    def is_source_closed(self) -> bool:
        return self._source_closed and self._bound_subject_id is not None

    def require_bound_subject(self, subject_id: str) -> None:
        if not self.is_source_closed or self._bound_subject_id != subject_id:
            raise ReporterResponseContractError(
                "profile subject must equal the exact subject bound by source-closed Reader evidence"
            )

    def require_biological_replicate_scopes(
        self,
        values: tuple[tuple[str, str | None], ...],
    ) -> None:
        """Require condition-scoped profile identities to equal the source declaration."""

        observed = tuple(sorted({(condition, value) for condition, value in values if value is not None}))
        contains_unknown = any(value is None for _, value in values)
        expected = self._declared_biological_replicate_scopes
        if expected:
            if contains_unknown or observed != expected:
                raise ReporterResponseContractError(
                    "profile condition-scoped biological-replicate identities must equal the source-closed "
                    "Reader binding"
                )
        elif observed:
            raise ReporterResponseContractError(
                "profile cannot invent biological-replicate identities when the Reader binding declares none"
            )
