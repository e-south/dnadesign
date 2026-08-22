"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/folding/src/assessment/api.py

Create-only structure-assessment orchestration.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from importlib.metadata import version
from pathlib import Path

from dnadesign.artifacts import CreateOnlyDirectoryPublication, PublicationError
from dnadesign.contracts.folding import (
    AssessmentProducerV1,
    StructureAssessmentPublicationV1,
    StructureAssessmentRecordV1,
    StructureAssessmentRequestV1,
)
from dnadesign.contracts.folding.secondary_structure_prediction_v2 import SecondaryStructurePredictionV2

from ..errors import FoldingConfigError, FoldingExecutionError
from ._limits import ARTIFACT_AGGREGATE_SIZE_LIMIT_BYTES, ARTIFACT_FILE_SIZE_LIMIT_BYTES
from .execution import prepare_prediction_request, run_worker
from .publication import (
    PublishedStructureAssessment,
    _AnchoredPublicationReader,
    artifact_digests,
    content_digest,
    model_json_bytes,
    verify_publication,
    write_model_json,
)

_MANIFEST = "manifest.json"
_REQUEST = "assessment-request.json"
_RECORD = "assessment-record.json"
_PREDICTION = "prediction/secondary_structure_prediction_v2.json"


def publish_structure_assessment(
    request: StructureAssessmentRequestV1,
    *,
    output_dir: str | Path,
) -> PublishedStructureAssessment:
    """Run one isolated assessment and atomically publish its evidence."""
    published: PublishedStructureAssessment | None = None
    try:
        publication = CreateOnlyDirectoryPublication.prepare(output_dir)
    except PublicationError as exc:
        raise FoldingConfigError(str(exc)) from exc
    try:
        stage = publication.stage
        request_content = write_model_json(stage / _REQUEST, request)
        request_digest = content_digest(request_content)
        low_level_path, target_content = prepare_prediction_request(stage, request)
        target_artifact_digest = content_digest(target_content)
        with _AnchoredPublicationReader.from_descriptor(publication.stage_descriptor) as reader:
            run_worker(
                low_level_path,
                stage / "prediction",
                artifact_root_descriptor=publication.stage_descriptor,
                timeout_seconds=request.policy.timeout_seconds,
            )
            worker_request_content = reader.read_bytes(
                "prediction/prediction-request.json",
                label="assessment worker request",
            )
            worker_request_digest = content_digest(worker_request_content)
            observed_target_content = reader.read_bytes(
                "assessment-target-sequence.json",
                label="assessment target sequence",
            )
            if content_digest(observed_target_content) != target_artifact_digest:
                raise FoldingExecutionError("Assessment target artifact changed during backend execution.")
            prediction_content = reader.read_bytes(
                _PREDICTION,
                label="assessment prediction",
            )
            prediction = SecondaryStructurePredictionV2.model_validate_json(prediction_content)
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
            record_content = model_json_bytes(record)
            reader.write_new_bytes(_RECORD, record_content, label="assessment record")
            inventory = artifact_digests(stage, reader=reader)
            manifest = StructureAssessmentPublicationV1(
                assessment_id=request.assessment_id,
                request_digest=request_digest,
                target_sequence_artifact_digest=target_artifact_digest,
                worker_request_digest=worker_request_digest,
                prediction_digest=prediction_digest,
                record_digest=content_digest(record_content),
                target_state_digest=request.target.state_digest,
                target_sequence_sha256=request.target.sequence_sha256,
                artifact_digests=inventory,
            )
            manifest_content = model_json_bytes(manifest)
            reader.write_new_bytes(_MANIFEST, manifest_content, label="assessment manifest")
            verified_stage = verify_publication(stage, allow_staging_owner=True, reader=reader)

            def verify_copied_descriptor(descriptor: int) -> None:
                with _AnchoredPublicationReader.from_descriptor(descriptor) as copied_reader:
                    copied = verify_publication(
                        stage,
                        allow_staging_owner=True,
                        reader=copied_reader,
                    )
                if copied != verified_stage:
                    raise PublicationError("Copied assessment does not match the verified staging assessment.")

            publication.publish(
                required_manifest=_MANIFEST,
                verify_copied_descriptor=verify_copied_descriptor,
                copy_file_size_limit_bytes=ARTIFACT_FILE_SIZE_LIMIT_BYTES,
                copy_aggregate_size_limit_bytes=ARTIFACT_AGGREGATE_SIZE_LIMIT_BYTES,
            )
        try:
            publication.assert_published_path_identity()
            published_descriptor = publication.duplicate_published_descriptor()
            try:
                with _AnchoredPublicationReader.from_descriptor(published_descriptor) as published_reader:
                    published = verify_publication(
                        publication.final,
                        reader=published_reader,
                    )
            finally:
                os.close(published_descriptor)
            if published != verified_stage:
                raise FoldingExecutionError("Published assessment does not match the verified staging assessment.")
            publication.assert_published_path_identity()
        except BaseException:
            publication.rollback()
            raise
    except PublicationError as exc:
        raise FoldingConfigError(str(exc)) from exc
    finally:
        publication.close()
    if published is None:
        raise FoldingExecutionError("Assessment publication completed without verified output.")
    return published


__all__ = ["publish_structure_assessment"]
