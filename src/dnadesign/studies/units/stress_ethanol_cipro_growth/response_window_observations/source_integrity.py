"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/source_integrity.py

 Revalidate the study-owned candidate-binding source.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from dnadesign.studies.core.reader_records import ReaderResolvedRecord

from .artifact_contract import ResponseWindowObservationArtifactError
from .artifact_io import file_sha256, read_json_object


def verify_candidate_binding_records(
    *,
    candidate_bindings_manifest_path: Path,
    candidate_bindings_path: Path,
) -> None:
    """Reject candidate-binding changes since the study preview was assembled."""

    binding = _manifest(candidate_bindings_manifest_path, label="candidate-binding manifest")
    record = binding.get("record")
    if not isinstance(record, dict):
        raise ResponseWindowObservationArtifactError("candidate-binding manifest record is malformed.")
    relative = record.get("path")
    expected = record.get("sha256")
    binding_root = candidate_bindings_manifest_path.parent.resolve()
    if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
        raise ResponseWindowObservationArtifactError("candidate-binding record identity or digest is invalid.")
    path = _confined(binding_root / relative, root=binding_root, label="candidate-binding record")
    if path != candidate_bindings_path.resolve():
        raise ResponseWindowObservationArtifactError("candidate-binding record path disagrees with verified preview.")
    if not path.is_file() or file_sha256(path) != expected:
        raise ResponseWindowObservationArtifactError("candidate-binding record drift detected after preview.")


def verify_reader_record_bytes(record_refs: Mapping[str, ReaderResolvedRecord]) -> None:
    """Reject dataframe-record changes since the canonical Reader resolution."""

    for name, record in record_refs.items():
        if record.path is None or record.content_digest is None or record.size_bytes is None:
            raise ResponseWindowObservationArtifactError(f"Reader {name!r} source is not a dataframe record.")
        try:
            content = record.path.read_bytes()
        except OSError as exc:
            raise ResponseWindowObservationArtifactError(f"Reader {name!r} source cannot be read: {exc}") from exc
        digest = "sha256:" + hashlib.sha256(content).hexdigest()
        if len(content) != record.size_bytes or digest != record.content_digest:
            raise ResponseWindowObservationArtifactError(f"Reader {name!r} source record drift detected.")


def _manifest(path: Path, *, label: str) -> dict[str, object]:
    try:
        return read_json_object(path, label=label)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ResponseWindowObservationArtifactError(f"could not revalidate {label}: {exc}") from exc


def _confined(path: Path, *, root: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ResponseWindowObservationArtifactError(f"{label} escapes its source bundle.") from exc
    return resolved


__all__ = ["verify_candidate_binding_records", "verify_reader_record_bytes"]
