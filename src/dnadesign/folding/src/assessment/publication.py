"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/folding/src/assessment/publication.py

Canonical serialization and replay verification for structure assessments.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pydantic import BaseModel

from dnadesign.contracts.folding import (
    AssessmentTargetSequenceV1,
    StructureAssessmentPublicationV1,
    StructureAssessmentRecordV1,
    StructureAssessmentRequestV1,
)
from dnadesign.contracts.folding.secondary_structure_prediction_v1 import SecondaryStructurePredictionV1

_MANIFEST = "manifest.json"
_STAGING_OWNER = ".dnadesign-publication-owner.json"


@dataclass(frozen=True, slots=True)
class PublishedStructureAssessment:
    """One verified create-only assessment publication."""

    manifest: StructureAssessmentPublicationV1
    request: StructureAssessmentRequestV1
    record: StructureAssessmentRecordV1


def model_json_bytes(model: BaseModel) -> bytes:
    """Return canonical indented JSON bytes for one contract model."""
    payload = model.model_dump(mode="json", by_alias=True)
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def content_digest(content: bytes) -> str:
    """Return the contract-form SHA-256 digest for exact bytes."""
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def write_model_json(path: Path, model: BaseModel) -> bytes:
    """Write and return one canonical contract representation."""
    content = model_json_bytes(model)
    path.write_bytes(content)
    return content


def artifact_digests(root: Path) -> dict[str, str]:
    """Inventory every non-manifest file in one staged publication."""
    return {
        path.relative_to(root).as_posix(): content_digest(path.read_bytes())
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in {_MANIFEST, _STAGING_OWNER}
    }


def _confined_file(root: Path, relative: str, *, label: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"{label} must stay inside the assessment publication.")
    path = root
    for part in relative_path.parts:
        path /= part
        if path.is_symlink():
            raise ValueError(f"{label} cannot use a symbolic link: {path}")
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes the assessment publication.") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} is missing.")
    return resolved


def verify_publication(
    root: Path,
    *,
    allow_staging_owner: bool = False,
) -> PublishedStructureAssessment:
    """Replay all byte and cross-object invariants in one publication."""
    manifest_path = _confined_file(root, _MANIFEST, label="assessment manifest")
    manifest = StructureAssessmentPublicationV1.model_validate_json(manifest_path.read_bytes())
    request_path = _confined_file(root, manifest.request_path, label="assessment request")
    target_sequence_path = _confined_file(
        root,
        manifest.target_sequence_path,
        label="assessment target sequence",
    )
    prediction_path = _confined_file(root, manifest.prediction_path, label="assessment prediction")
    record_path = _confined_file(root, manifest.record_path, label="assessment record")
    request_content = request_path.read_bytes()
    target_sequence_content = target_sequence_path.read_bytes()
    prediction_content = prediction_path.read_bytes()
    record_content = record_path.read_bytes()
    if content_digest(request_content) != manifest.request_digest:
        raise ValueError("Assessment request digest does not match the publication manifest.")
    if content_digest(target_sequence_content) != manifest.target_sequence_artifact_digest:
        raise ValueError("Assessment target-sequence artifact digest does not match the publication manifest.")
    if content_digest(prediction_content) != manifest.prediction_digest:
        raise ValueError("Assessment prediction digest does not match the publication manifest.")
    if content_digest(record_content) != manifest.record_digest:
        raise ValueError("Assessment record digest does not match the publication manifest.")
    _verify_artifact_inventory(
        root,
        manifest.artifact_digests,
        allow_staging_owner=allow_staging_owner,
    )
    request = StructureAssessmentRequestV1.model_validate_json(request_content)
    target_sequence = AssessmentTargetSequenceV1.model_validate_json(target_sequence_content)
    prediction = SecondaryStructurePredictionV1.model_validate_json(prediction_content)
    record = StructureAssessmentRecordV1.model_validate_json(record_content)
    if request.assessment_id != manifest.assessment_id or record.assessment_id != manifest.assessment_id:
        raise ValueError("Assessment identifiers do not agree across the publication.")
    if record.request_digest != manifest.request_digest or record.prediction_digest != manifest.prediction_digest:
        raise ValueError("Assessment record digests do not agree with the publication manifest.")
    if record.target != request.target or record.prediction != prediction:
        raise ValueError("Assessment record does not replay its request target and prediction.")
    if (
        target_sequence.sequence.id != request.target.sequence_id
        or f"sha256:{target_sequence.sequence.sha256}" != request.target.sequence_sha256
        or target_sequence.sequence.sequence != request.target.sequence
    ):
        raise ValueError("Assessment target artifact does not match the assessment request.")
    if (
        request.target.state_digest != manifest.target_state_digest
        or request.target.sequence_sha256 != manifest.target_sequence_sha256
    ):
        raise ValueError("Assessment target digests do not agree with the publication manifest.")
    return PublishedStructureAssessment(manifest=manifest, request=request, record=record)


def _verify_artifact_inventory(
    root: Path,
    expected: dict[str, str],
    *,
    allow_staging_owner: bool,
) -> None:
    actual_files: dict[str, str] = {}
    actual_directories: set[str] = set()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"Assessment artifact inventory cannot use a symbolic link: {relative}")
        if relative == _STAGING_OWNER and allow_staging_owner:
            if not path.is_file():
                raise ValueError("Assessment staging owner must be a regular file.")
            continue
        if path.is_dir():
            actual_directories.add(relative)
        elif path.is_file() and relative != _MANIFEST:
            actual_files[relative] = content_digest(path.read_bytes())
        elif relative != _MANIFEST:
            raise ValueError(f"Assessment artifact inventory contains an unsupported filesystem entry: {relative}")
    expected_directories = {
        parent.as_posix()
        for artifact_path in expected
        for parent in PurePosixPath(artifact_path).parents
        if parent.as_posix() != "."
    }
    if set(actual_files) != set(expected) or actual_directories != expected_directories:
        raise ValueError("Assessment artifact inventory does not match the publication contents.")
    mismatches = [path for path, digest in expected.items() if actual_files[path] != digest]
    if mismatches:
        raise ValueError(f"Assessment artifact inventory digest mismatch: {', '.join(sorted(mismatches))}")


def load_published_assessment(output_dir: str | Path) -> PublishedStructureAssessment:
    """Load and verify one create-only structure assessment publication."""
    path = Path(output_dir).expanduser()
    if ".." in path.parts:
        raise ValueError("Assessment publication directory cannot use parent traversal.")
    if not path.is_absolute():
        path = Path.cwd() / path
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"Assessment publication directory cannot use a symbolic link: {current}")
    root = path.resolve()
    if not root.is_dir():
        raise ValueError("Assessment publication directory is missing.")
    return verify_publication(root)


__all__ = ["PublishedStructureAssessment", "artifact_digests", "load_published_assessment"]
