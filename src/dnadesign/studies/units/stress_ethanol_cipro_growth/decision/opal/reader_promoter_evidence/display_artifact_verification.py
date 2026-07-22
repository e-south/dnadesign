"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/reader_promoter_evidence/display_artifact_verification.py

Verify content-addressed media in a Reader promoter-evidence display bundle.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath

from .contracts import (
    PROMOTER_EVIDENCE_ARTIFACT_IDS,
    PROMOTER_RESPONSE_SEMANTIC_KIND,
    READER_BUNDLE_SCHEMA_VERSION,
    READER_PROMOTER_EVIDENCE_MEDIA_DIR,
    ReaderPromoterEvidenceError,
)

_DISPLAY_ARTIFACT_FIELDS = {
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
    "source_manifest_sha256",
}
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")


def verify_display_artifact(
    value: object,
    *,
    manifest_root: Path,
    identity: tuple[str, str, str, str],
    source_manifest_sha256: str,
) -> str:
    if not isinstance(value, dict) or set(value) != _DISPLAY_ARTIFACT_FIELDS:
        raise ReaderPromoterEvidenceError(
            f"Reader display artifact fields must be exactly {sorted(_DISPLAY_ARTIFACT_FIELDS)}."
        )
    if (
        value["semantic_kind"] != PROMOTER_RESPONSE_SEMANTIC_KIND
        or value["kind"] != "reader_publication"
        or value["record_id"] != READER_BUNDLE_SCHEMA_VERSION
        or value["scope"] != "design_reduction"
        or value["exists"] is not True
        or value["source_manifest_sha256"] != source_manifest_sha256
    ):
        raise ReaderPromoterEvidenceError("Reader display artifact identity or source binding is invalid.")
    raw_path = value["path"]
    if not isinstance(raw_path, str) or not raw_path.strip() or "\\" in raw_path:
        raise ReaderPromoterEvidenceError("Reader display artifact path must be a relative POSIX path.")
    relative_path = PurePosixPath(raw_path)
    source_digest = source_manifest_sha256.removeprefix("sha256:")
    artifact_id = relative_path.name
    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
        or relative_path.parts != (READER_PROMOTER_EVIDENCE_MEDIA_DIR, source_digest, artifact_id)
    ):
        raise ReaderPromoterEvidenceError("Reader display artifact path is not content-addressed and confined.")
    artifact_path = (manifest_root / Path(*relative_path.parts)).resolve()
    try:
        artifact_path.relative_to(manifest_root)
    except ValueError as exc:
        raise ReaderPromoterEvidenceError("Reader display artifact path escapes its evidence bundle.") from exc
    if artifact_id not in PROMOTER_EVIDENCE_ARTIFACT_IDS:
        raise ReaderPromoterEvidenceError("Reader display artifact path has an unsupported media identity.")
    expected_media_type = "image/png" if artifact_id.endswith(".png") else "application/pdf"
    expected_label = f"{identity[2]}/{identity[1]}/{identity[3]}/{artifact_id}"
    expected_size = value["bytes"]
    if (
        value["path_label"] != expected_label
        or value["media_type"] != expected_media_type
        or isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 1
        or not is_sha256(value["sha256"])
    ):
        raise ReaderPromoterEvidenceError("Reader display artifact metadata is malformed.")
    if (
        not artifact_path.is_file()
        or artifact_path.stat().st_size != expected_size
        or _sha256(artifact_path) != value["sha256"]
    ):
        raise ReaderPromoterEvidenceError("Reader display artifact digest or size mismatch.")
    signature = artifact_path.read_bytes()[:8]
    if artifact_id.endswith(".pdf") and not signature.startswith(b"%PDF"):
        raise ReaderPromoterEvidenceError("Reader display PDF signature is invalid.")
    if artifact_id.endswith(".png") and signature != b"\x89PNG\r\n\x1a\n":
        raise ReaderPromoterEvidenceError("Reader display PNG signature is invalid.")
    return artifact_id


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


__all__ = ["is_sha256", "verify_display_artifact"]
