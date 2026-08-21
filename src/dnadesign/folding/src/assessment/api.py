"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/folding/src/assessment/api.py

Create-only structure-assessment orchestration.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

from dnadesign.artifacts import CreateOnlyDirectoryPublication, PublicationError
from dnadesign.contracts.folding import (
    AssessmentProducerV1,
    StructureAssessmentPublicationV1,
    StructureAssessmentRecordV1,
    StructureAssessmentRequestV1,
)
from dnadesign.contracts.folding.secondary_structure_prediction_v1 import SecondaryStructurePredictionV1

from ..errors import FoldingConfigError, FoldingExecutionError
from .execution import prepare_prediction_request, run_worker
from .publication import (
    PublishedStructureAssessment,
    artifact_digests,
    content_digest,
    load_published_assessment,
    verify_publication,
    write_model_json,
)

_MANIFEST = "manifest.json"
_REQUEST = "assessment-request.json"
_RECORD = "assessment-record.json"
_PREDICTION = "prediction/secondary_structure_prediction_v1.json"


def publish_structure_assessment(
    request: StructureAssessmentRequestV1,
    *,
    output_dir: str | Path,
) -> PublishedStructureAssessment:
    """Run one isolated assessment and atomically publish its evidence."""
    try:
        publication = CreateOnlyDirectoryPublication.prepare(output_dir)
    except PublicationError as exc:
        raise FoldingConfigError(str(exc)) from exc
    try:
        stage = publication.stage
        request_content = write_model_json(stage / _REQUEST, request)
        request_digest = content_digest(request_content)
        low_level_path = prepare_prediction_request(stage, request)
        run_worker(low_level_path, stage / "prediction", timeout_seconds=request.policy.timeout_seconds)
        prediction_path = stage / _PREDICTION
        prediction_content = prediction_path.read_bytes()
        prediction = SecondaryStructurePredictionV1.model_validate_json(prediction_content)
        if request.policy.required and prediction.status != "ok":
            errors = "; ".join(prediction.qa.errors or prediction.qa.warnings)
            raise FoldingExecutionError(errors or f"Required assessment ended with status {prediction.status}.")
        prediction_digest = content_digest(prediction_content)
        record = StructureAssessmentRecordV1(
            assessment_id=request.assessment_id,
            status=prediction.status,
            request_digest=request_digest,
            target=request.target,
            prediction_digest=prediction_digest,
            prediction=prediction,
            producer=AssessmentProducerV1(version=version("dnadesign")),
        )
        record_content = write_model_json(stage / _RECORD, record)
        inventory = artifact_digests(stage)
        manifest = StructureAssessmentPublicationV1(
            assessment_id=request.assessment_id,
            request_digest=request_digest,
            target_sequence_artifact_digest=inventory["assessment-target-sequence.json"],
            prediction_digest=prediction_digest,
            record_digest=content_digest(record_content),
            target_state_digest=request.target.state_digest,
            target_sequence_sha256=request.target.sequence_sha256,
            artifact_digests=inventory,
        )
        write_model_json(stage / _MANIFEST, manifest)
        verify_publication(stage)
        publication.publish(required_manifest=_MANIFEST)
    except PublicationError as exc:
        raise FoldingConfigError(str(exc)) from exc
    finally:
        publication.close()
    return load_published_assessment(output_dir)


__all__ = ["publish_structure_assessment"]
