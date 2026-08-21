"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/folding/assessment_v1.py

Digest-addressed advisory structure-assessment contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import re
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .secondary_structure_prediction_v1 import (
    SecondaryStructurePredictionRequestBackendV1,
    SecondaryStructurePredictionRequestPolicyV1,
    SecondaryStructurePredictionV1,
)

_SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
AssessmentStatus = Literal[
    "ok",
    "not_run",
    "error",
    "warning_optional_missing",
    "blocker_required_missing",
    "blocker_policy_unknown",
    "blocker_output_unwritable",
]


class AssessmentContractModel(BaseModel):
    """Strict immutable base for assessment authority records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _not_blank(value: str, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} cannot be empty.")
    return text


class AssessmentIntendedPairV1(AssessmentContractModel):
    """One expected zero-based pair in the assessed molecular state."""

    left: int = Field(ge=0)
    right: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_order(self) -> AssessmentIntendedPairV1:
        if self.right <= self.left:
            raise ValueError("intended pair right coordinate must be greater than left.")
        return self


class AssessmentTargetV1(AssessmentContractModel):
    """Exact molecular state submitted for advisory structure assessment."""

    contract: Literal["assessment_target_v1"] = "assessment_target_v1"
    schema_version: Literal[1] = 1
    state_id: str
    state_type: str
    state_schema: str
    state_digest: str = Field(pattern=_SHA256_PATTERN)
    sequence_id: str
    sequence_sha256: str = Field(pattern=_SHA256_PATTERN)
    sequence: str = Field(min_length=1)
    alphabet: Literal["dna"] = "dna"
    strandedness: Literal["single", "double", "not_asserted"]
    topology: Literal["linear", "circular", "not_asserted"]
    intended_pairs: tuple[AssessmentIntendedPairV1, ...] = ()

    @field_validator("state_id", "state_type", "state_schema", "sequence_id")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        return _not_blank(value, label="assessment target identity")

    @field_validator("sequence", mode="before")
    @classmethod
    def normalize_sequence(cls, value: object) -> str:
        if not isinstance(value, str) or not value or set(value.upper()) - set("ACGT"):
            raise ValueError("assessment target sequence must use non-empty exact DNA.")
        return value.upper()

    @model_validator(mode="after")
    def validate_target(self) -> AssessmentTargetV1:
        digest = f"sha256:{hashlib.sha256(self.sequence.encode()).hexdigest()}"
        if self.sequence_sha256 != digest:
            raise ValueError("sequence_sha256 must match the assessment target sequence.")
        coordinates = {(pair.left, pair.right) for pair in self.intended_pairs}
        if len(coordinates) != len(self.intended_pairs):
            raise ValueError("assessment target intended pairs must be unique.")
        if any(pair.right >= len(self.sequence) for pair in self.intended_pairs):
            raise ValueError("assessment target intended pair coordinate exceeds sequence length.")
        return self


class StructureAssessmentPolicyV1(SecondaryStructurePredictionRequestPolicyV1):
    """Execution policy for one isolated assessment."""

    timeout_seconds: float = Field(default=60.0, ge=0.1, le=600.0)


class StructureAssessmentRequestV1(AssessmentContractModel):
    """One backend request against an exact molecular state."""

    contract: Literal["structure_assessment_request_v1"] = "structure_assessment_request_v1"
    schema_version: Literal[1] = 1
    assessment_id: str
    target: AssessmentTargetV1
    backend: SecondaryStructurePredictionRequestBackendV1
    policy: StructureAssessmentPolicyV1 = Field(default_factory=StructureAssessmentPolicyV1)

    @field_validator("assessment_id")
    @classmethod
    def validate_assessment_id(cls, value: str) -> str:
        return _not_blank(value, label="assessment_id")


class AssessmentProducerV1(AssessmentContractModel):
    """Versioned producer identity for an assessment record."""

    name: Literal["dnadesign.folding"] = "dnadesign.folding"
    version: str

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        return _not_blank(value, label="assessment producer version")


class StructureAssessmentRecordV1(AssessmentContractModel):
    """Immutable advisory record for one exact target and prediction."""

    contract: Literal["structure_assessment_record_v1"] = "structure_assessment_record_v1"
    schema_version: Literal[1] = 1
    assessment_id: str
    authority: Literal["advisory"] = "advisory"
    status: AssessmentStatus
    request_digest: str = Field(pattern=_SHA256_PATTERN)
    target: AssessmentTargetV1
    prediction_digest: str = Field(pattern=_SHA256_PATTERN)
    prediction: SecondaryStructurePredictionV1
    producer: AssessmentProducerV1

    @field_validator("assessment_id", "status")
    @classmethod
    def validate_identity(cls, value: str) -> str:
        return _not_blank(value, label="assessment record field")

    @model_validator(mode="after")
    def validate_prediction_identity(self) -> StructureAssessmentRecordV1:
        prediction = self.prediction
        target = self.target
        if self.assessment_id != prediction.prediction_id:
            raise ValueError("assessment_id must match the prediction identifier.")
        if self.status != prediction.status:
            raise ValueError("assessment status must match the prediction status.")
        if (
            prediction.input.sequence_id != target.sequence_id
            or f"sha256:{prediction.input.sequence_sha256}" != target.sequence_sha256
            or prediction.input.length != len(target.sequence)
        ):
            raise ValueError("assessment prediction input must match the exact target sequence.")
        return self


class StructureAssessmentPublicationV1(AssessmentContractModel):
    """Digest manifest for one create-only assessment publication."""

    contract: Literal["structure_assessment_publication_v1"] = "structure_assessment_publication_v1"
    schema_version: Literal[1] = 1
    assessment_id: str
    request_path: Literal["assessment-request.json"] = "assessment-request.json"
    request_digest: str = Field(pattern=_SHA256_PATTERN)
    target_sequence_path: Literal["assessment-target-sequence.json"] = "assessment-target-sequence.json"
    target_sequence_artifact_digest: str = Field(pattern=_SHA256_PATTERN)
    prediction_path: Literal["prediction/secondary_structure_prediction_v1.json"] = (
        "prediction/secondary_structure_prediction_v1.json"
    )
    prediction_digest: str = Field(pattern=_SHA256_PATTERN)
    record_path: Literal["assessment-record.json"] = "assessment-record.json"
    record_digest: str = Field(pattern=_SHA256_PATTERN)
    target_state_digest: str = Field(pattern=_SHA256_PATTERN)
    target_sequence_sha256: str = Field(pattern=_SHA256_PATTERN)
    artifact_digests: dict[str, str] = Field(min_length=1)

    @field_validator("assessment_id")
    @classmethod
    def validate_assessment_id(cls, value: str) -> str:
        return _not_blank(value, label="assessment publication id")

    @field_validator("artifact_digests")
    @classmethod
    def validate_artifact_digests(cls, value: dict[str, str]) -> dict[str, str]:
        for artifact_path, digest in value.items():
            path = PurePosixPath(artifact_path)
            if (
                not artifact_path
                or "\\" in artifact_path
                or path.is_absolute()
                or "." in path.parts
                or ".." in path.parts
            ):
                raise ValueError("assessment artifact paths must be normalized relative POSIX paths.")
            if re.fullmatch(_SHA256_PATTERN, digest) is None:
                raise ValueError("assessment artifact digests must be sha256-prefixed lowercase hex.")
        return value

    @model_validator(mode="after")
    def validate_named_artifacts(self) -> StructureAssessmentPublicationV1:
        named_artifacts = {
            self.request_path: self.request_digest,
            self.target_sequence_path: self.target_sequence_artifact_digest,
            self.prediction_path: self.prediction_digest,
            self.record_path: self.record_digest,
        }
        if any(self.artifact_digests.get(path) != digest for path, digest in named_artifacts.items()):
            raise ValueError("named assessment artifacts must agree with the exhaustive artifact inventory.")
        return self


__all__ = [
    "AssessmentIntendedPairV1",
    "AssessmentProducerV1",
    "AssessmentStatus",
    "AssessmentTargetV1",
    "StructureAssessmentPolicyV1",
    "StructureAssessmentPublicationV1",
    "StructureAssessmentRecordV1",
    "StructureAssessmentRequestV1",
]
