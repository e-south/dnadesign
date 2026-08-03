"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/evidence_projection/provenance.py

Non-authoritative Reader lineage projection for offline profile evaluation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from dnadesign.studies.core.reader_records import ReaderRecordInputEvidence, ReaderRecordProducer

from ..._contract_values import sha256_digest
from ...policy import ReporterResponseObservationPolicy
from ._values import required_text


class ProfileProvenanceSource(Protocol):
    provenance: ProfileProvenanceProjection
    observation_policy: ReporterResponseObservationPolicy


@dataclass(frozen=True, slots=True)
class ProfileProvenanceProjection:
    """Serialized provenance identity without source-closure authority."""

    raw_design_id: str | None
    raw_assay_subject_id: str | None
    reader_experiment_id: str
    reader_protocol_id: str
    reader_record_id: str
    reader_record_kind: str
    reader_record_revision: int
    reader_record_revision_digest: str
    reader_record_config_digest: str
    reader_record_producer_config_digest: str
    reader_record_producer: ReaderRecordProducer
    reader_record_inputs: tuple[ReaderRecordInputEvidence, ...]
    reader_record_content_digest: str
    reader_record_schema_version: int
    reader_record_contract_id: str
    reader_record_path: str
    evidence_binding_artifact_id: str
    evidence_binding_artifact_digest: str

    def __post_init__(self) -> None:
        if self.raw_design_id is None and self.raw_assay_subject_id is None:
            raise ValueError("provenance requires at least one raw Reader identity")
        for name in ("raw_design_id", "raw_assay_subject_id"):
            value = getattr(self, name)
            if value is not None:
                required_text(value, label=name)
        for name in (
            "reader_experiment_id",
            "reader_protocol_id",
            "reader_record_path",
        ):
            required_text(getattr(self, name), label=name)
        if self.reader_record_id != "sample_measurements/df":
            raise ValueError("reader_record_id must equal sample_measurements/df")
        if self.reader_record_kind != "dataframe_artifact":
            raise ValueError("reader_record_kind must equal dataframe_artifact")
        if self.reader_record_schema_version != 6:
            raise ValueError("reader_record_schema_version must equal 6")
        if self.reader_record_contract_id != "plate_reader.annotated.v1":
            raise ValueError("reader_record_contract_id must equal plate_reader.annotated.v1")
        if type(self.reader_record_revision) is not int or self.reader_record_revision < 1:
            raise ValueError("reader_record_revision must be positive")
        for name in (
            "reader_record_revision_digest",
            "reader_record_config_digest",
            "reader_record_producer_config_digest",
            "reader_record_content_digest",
            "evidence_binding_artifact_digest",
        ):
            sha256_digest(getattr(self, name), field_name=name)
        if not isinstance(self.reader_record_producer, ReaderRecordProducer):
            raise ValueError("reader_record_producer must be typed Reader provenance")
        if not isinstance(self.reader_record_inputs, tuple) or not all(
            isinstance(item, ReaderRecordInputEvidence) for item in self.reader_record_inputs
        ):
            raise ValueError("reader_record_inputs must be typed Reader provenance")
        record_path = Path(self.reader_record_path)
        if record_path.is_absolute() or ".." in record_path.parts:
            raise ValueError("reader_record_path must be outputs-relative")


def profile_source_identity_projection(profile: ProfileProvenanceSource) -> dict[str, object]:
    """Return the source coordinate without claiming current authenticity."""

    provenance = profile.provenance
    return {
        "raw_design_id": provenance.raw_design_id,
        "raw_assay_subject_id": provenance.raw_assay_subject_id,
        "reader_experiment_id": provenance.reader_experiment_id,
        "reader_protocol_id": provenance.reader_protocol_id,
        "reader_record_id": provenance.reader_record_id,
        "reader_record_kind": provenance.reader_record_kind,
        "reader_record_revision": provenance.reader_record_revision,
        "reader_record_revision_digest": provenance.reader_record_revision_digest,
        "reader_record_config_digest": provenance.reader_record_config_digest,
        "reader_record_producer_config_digest": provenance.reader_record_producer_config_digest,
        "reader_record_producer": provenance.reader_record_producer.to_dict(),
        "reader_record_inputs": [item.to_dict() for item in provenance.reader_record_inputs],
        "reader_record_content_digest": provenance.reader_record_content_digest,
        "reader_record_schema_version": provenance.reader_record_schema_version,
        "reader_record_contract_id": provenance.reader_record_contract_id,
        "reader_record_path": provenance.reader_record_path,
        "evidence_binding_artifact_id": provenance.evidence_binding_artifact_id,
        "evidence_binding_artifact_digest": provenance.evidence_binding_artifact_digest,
        "observation_policy_identity": profile.observation_policy.digest,
    }


__all__ = ["ProfileProvenanceProjection", "profile_source_identity_projection"]
