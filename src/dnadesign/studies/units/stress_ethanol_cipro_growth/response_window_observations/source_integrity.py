"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/response_window_observations/source_integrity.py

Revalidate every file bound by the verified Reader and candidate bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from .artifact_contract import ResponseWindowObservationArtifactError
from .artifact_io import file_sha256, read_json_object


def verify_source_bundle_records(
    *,
    reader_manifest_path: Path,
    candidate_bindings_manifest_path: Path,
    candidate_bindings_path: Path,
) -> None:
    """Reject any source-record change since the study preview was assembled."""

    reader = _manifest(reader_manifest_path, label="Reader manifest")
    artifacts = reader.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ResponseWindowObservationArtifactError("Reader manifest lacks its artifact inventory.")
    reader_root = reader_manifest_path.parent.resolve()
    for artifact_id, raw in artifacts.items():
        if not isinstance(artifact_id, str) or not isinstance(raw, dict) or set(raw) != {"path", "bytes", "sha256"}:
            raise ResponseWindowObservationArtifactError(f"Reader artifact {artifact_id!r} metadata is malformed.")
        relative = raw["path"]
        size = raw["bytes"]
        expected = raw["sha256"]
        if relative != artifact_id:
            raise ResponseWindowObservationArtifactError(
                f"Reader artifact {artifact_id!r} path disagrees with its manifest identity."
            )
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ResponseWindowObservationArtifactError(f"Reader artifact {artifact_id!r} byte count is invalid.")
        if not isinstance(expected, str) or not expected.startswith("sha256:") or len(expected) != 71:
            raise ResponseWindowObservationArtifactError(f"Reader artifact {artifact_id!r} digest is invalid.")
        path = _confined(reader_root / artifact_id, root=reader_root, label=f"Reader artifact {artifact_id!r}")
        if not path.is_file() or path.stat().st_size != size or f"sha256:{file_sha256(path)}" != expected:
            raise ResponseWindowObservationArtifactError(
                f"Reader artifact {artifact_id!r} drift detected after preview."
            )

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


__all__ = ["verify_source_bundle_records"]
