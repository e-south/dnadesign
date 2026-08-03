"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/reader_promoter_evidence/display_artifact_verification.py

Verify staged media projected from one exact Reader plot record revision.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath
from typing import Mapping

from .contracts import (
    PROMOTER_RESPONSE_SEMANTIC_KIND,
    READER_EVENT_WINDOW_DIAGNOSTIC_RECORD_ID,
    READER_PROMOTER_EVIDENCE_MEDIA_DIR,
    ReaderPromoterEvidenceError,
    canonical_json_sha256,
)

_FIELDS = {
    "semantic_kind",
    "kind",
    "record_id",
    "scope",
    "path",
    "path_label",
    "exists",
    "media_type",
    "bytes",
    "sha256",
    "source_record_revision_digest",
    "source_file_path",
    "source_receipt_sha256",
}
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")


def verify_display_artifact(
    value: object,
    *,
    manifest_root: Path,
    identity: tuple[str, str, str, str],
    response_source: Mapping[str, object],
) -> None:
    """Verify content addressing, source receipt, and staged media bytes."""

    if not isinstance(value, dict) or set(value) != _FIELDS:
        raise ReaderPromoterEvidenceError(f"Reader display artifact fields must be exactly {sorted(_FIELDS)}.")
    records = response_source["records"]
    if not isinstance(records, Mapping):  # pragma: no cover - source verification runs first
        raise ReaderPromoterEvidenceError("Reader response source records are malformed.")
    diagnostic = records["diagnostic"]
    if not isinstance(diagnostic, Mapping):  # pragma: no cover - source verification runs first
        raise ReaderPromoterEvidenceError("Reader diagnostic source is malformed.")
    revision_digest = str(diagnostic["revision_digest"])
    if (
        value["semantic_kind"] != PROMOTER_RESPONSE_SEMANTIC_KIND
        or value["kind"] != "reader_record_projection"
        or value["record_id"] != READER_EVENT_WINDOW_DIAGNOSTIC_RECORD_ID
        or value["scope"] != "design_reduction"
        or value["exists"] is not True
        or value["source_record_revision_digest"] != revision_digest
        or value["source_receipt_sha256"] != canonical_json_sha256(response_source)
    ):
        raise ReaderPromoterEvidenceError("Reader display artifact identity or source binding is invalid.")
    source_path = _relative_path(value["source_file_path"], field="source_file_path")
    evidence = diagnostic["file_evidence"]
    matches = [item for item in evidence if isinstance(item, Mapping) and item.get("path") == source_path]
    if len(matches) != 1:
        raise ReaderPromoterEvidenceError("Reader display artifact must bind one exact diagnostic file-evidence row.")
    source_evidence = matches[0]
    expected_size = source_evidence.get("size_bytes")
    expected_digest = source_evidence.get("content_digest")
    if value["bytes"] != expected_size or value["sha256"] != expected_digest:
        raise ReaderPromoterEvidenceError("Reader display artifact disagrees with Reader file evidence.")
    artifact_name = Path(source_path).name
    relative = PurePosixPath(_relative_path(value["path"], field="path"))
    expected_parts = (
        READER_PROMOTER_EVIDENCE_MEDIA_DIR,
        revision_digest.removeprefix("sha256:"),
        artifact_name,
    )
    if relative.parts != expected_parts:
        raise ReaderPromoterEvidenceError("Reader display artifact path is not revision-addressed and confined.")
    artifact_path = (manifest_root / Path(*relative.parts)).resolve()
    try:
        artifact_path.relative_to(manifest_root)
    except ValueError as exc:
        raise ReaderPromoterEvidenceError("Reader display artifact path escapes its manifest root.") from exc
    suffix = artifact_path.suffix.lower()
    expected_media_type = "image/png" if suffix == ".png" else "application/pdf" if suffix == ".pdf" else None
    expected_label = f"{identity[2]}/{identity[1]}/{identity[3]}/{artifact_name}"
    if value["media_type"] != expected_media_type or value["path_label"] != expected_label:
        raise ReaderPromoterEvidenceError("Reader display artifact label or media type is invalid.")
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 1
        or not is_sha256(expected_digest)
        or not artifact_path.is_file()
        or artifact_path.stat().st_size != expected_size
        or _sha256(artifact_path) != expected_digest
    ):
        raise ReaderPromoterEvidenceError("Reader display artifact digest or size mismatch.")
    signature = artifact_path.read_bytes()[:8]
    if suffix == ".png" and signature != b"\x89PNG\r\n\x1a\n":
        raise ReaderPromoterEvidenceError("Reader display PNG signature is invalid.")
    if suffix == ".pdf" and not signature.startswith(b"%PDF"):
        raise ReaderPromoterEvidenceError("Reader display PDF signature is invalid.")


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _relative_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\\" in value:
        raise ReaderPromoterEvidenceError(f"Reader display {field} must be a relative POSIX path.")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
        raise ReaderPromoterEvidenceError(f"Reader display {field} must be a confined relative POSIX path.")
    return value


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["is_sha256", "verify_display_artifact"]
