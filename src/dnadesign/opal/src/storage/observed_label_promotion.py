"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/storage/observed_label_promotion.py

Verification for study-published, manifest-pinned observed-label artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import pyarrow.parquet as pq

from ..core.utils import OpalError, file_sha256
from .candidate_exclusion_projection import (
    CandidateExclusionSetBinding,
    verify_candidate_exclusion_projection,
)
from .candidate_snapshot import verify_candidate_snapshot

OBSERVED_LABEL_PROMOTION_SCHEMA_VERSION = "opal.observed_label_promotion.v1"
_MANIFEST_FIELDS = {
    "schema_version",
    "campaign_slug",
    "study_id",
    "y_space",
    "study_provenance",
    "candidate_exclusion_projection",
    "candidate_artifact",
    "label_artifact",
}
_STUDY_PROVENANCE_FIELDS = {"schema_id", "path", "sha256"}
_LABEL_ARTIFACT_FIELDS = {"path", "sha256", "row_count"}


class _DuplicateJsonKeyError(ValueError):
    pass


@dataclass(frozen=True)
class ObservedLabelPromotionBinding:
    """Expected identity for one study-published observed-label snapshot."""

    dataset_root: Path
    manifest_path: str
    label_path: str
    campaign_slug: str
    study_id: str
    y_space: str
    candidate_path: str = "records.parquet"
    candidate_id_column: str = "id"
    candidate_x_column: str | None = None
    candidate_root: Path | None = None
    candidate_exclusion_sets: tuple[CandidateExclusionSetBinding, ...] | None = None


@dataclass(frozen=True)
class VerifiedObservedLabelPromotion:
    """Resolved promotion identity after manifest and artifact verification."""

    manifest_path: Path
    manifest_sha256: str
    label_path: Path
    label_sha256: str
    row_count: int
    campaign_slug: str
    study_id: str
    y_space: str
    study_provenance_schema_id: str
    study_provenance_path: Path
    study_provenance_sha256: str
    candidate_exclusion_set_id: str
    candidate_exclusion_entries_sha256: str
    candidate_exclusion_entry_count: int
    candidate_path: Path
    candidate_sha256: str
    candidate_row_count: int
    candidate_columns: tuple[str, ...]
    candidate_schema_sha256: str


def _resolve_dataset_relative(
    dataset_root: Path,
    value: str | Path,
    *,
    field: str,
) -> tuple[str, Path]:
    raw = str(value).strip()
    posix_path = PurePosixPath(raw)
    windows_path = PureWindowsPath(raw)
    if not raw or "\\" in raw or posix_path.is_absolute() or windows_path.is_absolute() or ".." in posix_path.parts:
        raise OpalError(f"Observed-label promotion {field} must remain within the USR dataset root.")
    root = Path(dataset_root).resolve()
    resolved = (root / Path(*posix_path.parts)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise OpalError(f"Observed-label promotion {field} must remain within the USR dataset root.") from exc
    return posix_path.as_posix(), resolved


def _required_string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise OpalError(f"Observed-label promotion manifest field {field!r} must be a non-empty string.")
    return value.strip()


def _require_match(*, field: str, actual: str, expected: str) -> None:
    if actual != expected:
        raise OpalError(f"Observed-label promotion {field} mismatch: expected {expected!r}, found {actual!r}.")


def _read_manifest(path: Path) -> tuple[dict[str, Any], str]:
    if not path.exists():
        raise OpalError(f"Observed-label promotion manifest not found: {path}")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_json_object)
    except _DuplicateJsonKeyError as exc:
        raise OpalError(f"Observed-label promotion manifest has {exc}: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OpalError(f"Failed to read observed-label promotion manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise OpalError(f"Observed-label promotion manifest must be a JSON object: {path}")
    return payload, sha256(raw).hexdigest()


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise _DuplicateJsonKeyError(f"duplicate JSON key {key!r}")
        payload[key] = value
    return payload


def verify_observed_label_promotion(
    binding: ObservedLabelPromotionBinding,
) -> VerifiedObservedLabelPromotion:
    """Verify one immutable label snapshot against its campaign binding."""

    _, manifest_path = _resolve_dataset_relative(
        binding.dataset_root,
        binding.manifest_path,
        field="manifest_path",
    )
    expected_label_relative, expected_label_path = _resolve_dataset_relative(
        binding.dataset_root,
        binding.label_path,
        field="label_path",
    )
    payload, manifest_sha256 = _read_manifest(manifest_path)
    if set(payload) != _MANIFEST_FIELDS:
        raise OpalError(f"Observed-label promotion manifest fields must be exactly {sorted(_MANIFEST_FIELDS)}.")

    schema_version = _required_string(payload, "schema_version")
    _require_match(
        field="schema_version",
        actual=schema_version,
        expected=OBSERVED_LABEL_PROMOTION_SCHEMA_VERSION,
    )
    campaign_slug = _required_string(payload, "campaign_slug")
    study_id = _required_string(payload, "study_id")
    y_space = _required_string(payload, "y_space")
    _require_match(field="campaign_slug", actual=campaign_slug, expected=str(binding.campaign_slug))
    _require_match(field="study_id", actual=study_id, expected=str(binding.study_id))
    _require_match(field="y_space", actual=y_space, expected=str(binding.y_space))

    study_provenance = payload.get("study_provenance")
    if not isinstance(study_provenance, dict):
        raise OpalError("Observed-label promotion manifest field 'study_provenance' must be a JSON object.")
    if set(study_provenance) != _STUDY_PROVENANCE_FIELDS:
        raise OpalError(
            f"Observed-label promotion study_provenance fields must be exactly {sorted(_STUDY_PROVENANCE_FIELDS)}."
        )
    provenance_schema_id = _required_string(study_provenance, "schema_id")
    provenance_relative_raw = _required_string(study_provenance, "path")
    _, provenance_path = _resolve_dataset_relative(
        binding.dataset_root,
        provenance_relative_raw,
        field="study_provenance.path",
    )
    provenance_expected_sha256 = _required_string(study_provenance, "sha256")
    if re.fullmatch(r"[0-9a-f]{64}", provenance_expected_sha256) is None:
        raise OpalError("Observed-label promotion study_provenance.sha256 must be a lowercase SHA-256 digest.")
    if not provenance_path.is_file():
        raise OpalError(f"Observed-label promotion study provenance not found: {provenance_path}")
    try:
        provenance_actual_sha256 = file_sha256(provenance_path)
    except OSError as exc:
        raise OpalError(f"Failed to hash observed-label study provenance {provenance_path}: {exc}") from exc
    if provenance_actual_sha256 != provenance_expected_sha256:
        raise OpalError(
            "Observed-label promotion study provenance SHA-256 mismatch: "
            f"expected {provenance_expected_sha256}, found {provenance_actual_sha256}."
        )

    candidate_exclusions = verify_candidate_exclusion_projection(
        payload.get("candidate_exclusion_projection"),
        configured_sets=binding.candidate_exclusion_sets,
    )

    candidate = verify_candidate_snapshot(
        payload.get("candidate_artifact"),
        root=binding.dataset_root if binding.candidate_root is None else binding.candidate_root,
        expected_path=binding.candidate_path,
        id_column=binding.candidate_id_column,
        x_column=binding.candidate_x_column,
    )

    artifact = payload.get("label_artifact")
    if not isinstance(artifact, dict):
        raise OpalError("Observed-label promotion manifest field 'label_artifact' must be a JSON object.")
    if set(artifact) != _LABEL_ARTIFACT_FIELDS:
        raise OpalError(
            f"Observed-label promotion label_artifact fields must be exactly {sorted(_LABEL_ARTIFACT_FIELDS)}."
        )
    artifact_relative_raw = _required_string(artifact, "path")
    artifact_relative, artifact_path = _resolve_dataset_relative(
        binding.dataset_root,
        artifact_relative_raw,
        field="label_artifact.path",
    )
    _require_match(field="label_artifact.path", actual=artifact_relative, expected=expected_label_relative)
    if artifact_path != expected_label_path:
        raise OpalError(
            "Observed-label promotion label_artifact.path does not resolve to the configured label sidecar."
        )

    expected_sha256 = _required_string(artifact, "sha256")
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise OpalError("Observed-label promotion label_artifact.sha256 must be a lowercase SHA-256 digest.")
    expected_row_count = artifact.get("row_count")
    if isinstance(expected_row_count, bool) or not isinstance(expected_row_count, int) or expected_row_count < 0:
        raise OpalError("Observed-label promotion label_artifact.row_count must be a non-negative integer.")

    if not artifact_path.exists():
        raise OpalError(f"Observed-label promotion label artifact not found: {artifact_path}")
    try:
        actual_sha256 = file_sha256(artifact_path)
    except OSError as exc:
        raise OpalError(f"Failed to hash observed-label promotion artifact {artifact_path}: {exc}") from exc
    if actual_sha256 != expected_sha256:
        raise OpalError(
            "Observed-label promotion label artifact SHA-256 mismatch: "
            f"expected {expected_sha256}, found {actual_sha256}."
        )
    try:
        actual_row_count = int(pq.ParquetFile(artifact_path).metadata.num_rows)
    except Exception as exc:
        raise OpalError(f"Failed to inspect observed-label promotion artifact {artifact_path}: {exc}") from exc
    if actual_row_count != expected_row_count:
        raise OpalError(
            "Observed-label promotion label_artifact.row_count mismatch: "
            f"expected {expected_row_count}, found {actual_row_count}."
        )

    return VerifiedObservedLabelPromotion(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        label_path=artifact_path,
        label_sha256=actual_sha256,
        row_count=actual_row_count,
        campaign_slug=campaign_slug,
        study_id=study_id,
        y_space=y_space,
        study_provenance_schema_id=provenance_schema_id,
        study_provenance_path=provenance_path,
        study_provenance_sha256=provenance_actual_sha256,
        candidate_exclusion_set_id=candidate_exclusions.exclusion_set_id,
        candidate_exclusion_entries_sha256=candidate_exclusions.entries_sha256,
        candidate_exclusion_entry_count=candidate_exclusions.entry_count,
        candidate_path=candidate.path,
        candidate_sha256=candidate.sha256,
        candidate_row_count=candidate.row_count,
        candidate_columns=candidate.columns,
        candidate_schema_sha256=candidate.schema_sha256,
    )
