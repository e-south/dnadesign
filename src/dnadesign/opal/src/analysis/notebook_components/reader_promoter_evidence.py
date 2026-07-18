"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/reader_promoter_evidence.py

Render-time integrity checks for static Reader promoter evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from dnadesign.opal.api.reader_evidence import (
    ReaderEvidenceManifestAdapterError,
    parse_reader_evidence_manifest_adapter,
)

PROMOTER_RESPONSE_EVIDENCE_SEMANTIC_KIND = "promoter_response_evidence"
PROMOTER_EVIDENCE_BUNDLE_RECORD_ID = "reader.response_window.promoter_evidence_bundle.v4"
READER_PROMOTER_EVIDENCE_MAX_BYTES = 32 * 1024 * 1024

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_MEDIA_TYPES = {
    "promoter_evidence.png": "image/png",
    "promoter_evidence.pdf": "application/pdf",
}


class ReaderPromoterEvidenceIntegrityError(ValueError):
    """Raised when a static promoter-evidence row no longer matches its source."""


def is_reader_promoter_evidence_artifact(row: Mapping[str, Any]) -> bool:
    """Return true only for the metric-neutral promoter evidence semantic kind."""

    return str(row.get("semantic_kind") or "").strip() == PROMOTER_RESPONSE_EVIDENCE_SEMANTIC_KIND


def verify_reader_promoter_evidence_artifact(row: Mapping[str, Any]) -> Path:
    """Verify size, digests, signatures, and exact source-manifest binding."""

    if not is_reader_promoter_evidence_artifact(row):
        raise ReaderPromoterEvidenceIntegrityError("Artifact is not Reader promoter-response evidence.")
    if (
        row.get("kind") != "reader_publication"
        or row.get("artifact_record_id") != PROMOTER_EVIDENCE_BUNDLE_RECORD_ID
        or row.get("scope") != "design_reduction"
    ):
        raise ReaderPromoterEvidenceIntegrityError("Promoter-evidence publication identity is invalid.")
    evidence_role = str(row.get("evidence_role") or "")
    claim_status = str(row.get("claim_status") or "")
    if evidence_role != "display_only" or claim_status not in {"objective_neutral", "screen_only"}:
        raise ReaderPromoterEvidenceIntegrityError("Promoter evidence must remain display-only with a supported claim.")
    artifact_path, manifest_path = _staged_artifact_path(
        row.get("path"),
        manifest_path=row.get("manifest_path"),
        source_manifest_sha256=row.get("source_manifest_sha256"),
    )
    expected_media_type = _MEDIA_TYPES.get(artifact_path.name)
    if expected_media_type is None or row.get("media_type") != expected_media_type:
        raise ReaderPromoterEvidenceIntegrityError("Promoter artifact filename or media type is invalid.")
    expected_size = row.get("bytes")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 1:
        raise ReaderPromoterEvidenceIntegrityError("Promoter artifact bytes must be a positive integer.")
    if expected_size > READER_PROMOTER_EVIDENCE_MAX_BYTES:
        raise ReaderPromoterEvidenceIntegrityError(
            f"Promoter artifact exceeds the {READER_PROMOTER_EVIDENCE_MAX_BYTES}-byte render ceiling."
        )
    if not artifact_path.is_file() or artifact_path.stat().st_size != expected_size:
        raise ReaderPromoterEvidenceIntegrityError("Promoter artifact size no longer matches its display manifest.")
    _verify_digest(artifact_path, expected=row.get("sha256"), label="artifact")
    _verify_display_identity(row)
    _verify_publication_manifest_binding(row, manifest_path=manifest_path)
    with artifact_path.open("rb") as stream:
        signature = stream.read(8)
    if artifact_path.suffix == ".png" and signature != b"\x89PNG\r\n\x1a\n":
        raise ReaderPromoterEvidenceIntegrityError("Promoter PNG signature is invalid.")
    if artifact_path.suffix == ".pdf" and not signature.startswith(b"%PDF"):
        raise ReaderPromoterEvidenceIntegrityError("Promoter PDF signature is invalid.")
    return artifact_path


def _staged_artifact_path(
    value: object,
    *,
    manifest_path: object,
    source_manifest_sha256: object,
) -> tuple[Path, Path]:
    manifest_raw = str(manifest_path or "")
    manifest = Path(manifest_raw).expanduser()
    if (
        not manifest_raw
        or manifest_raw != str(manifest)
        or not manifest.is_absolute()
        or manifest.resolve() != manifest
    ):
        raise ReaderPromoterEvidenceIntegrityError("Promoter display manifest path must be exact and absolute.")
    source_digest = str(source_manifest_sha256 or "")
    if _SHA256.fullmatch(source_digest) is None:
        raise ReaderPromoterEvidenceIntegrityError("Promoter Reader source-manifest SHA-256 is invalid.")
    raw = str(value or "")
    relative = PurePosixPath(raw)
    artifact_id = relative.name
    expected_parts = ("reader_evidence_media", source_digest.removeprefix("sha256:"), artifact_id)
    if not raw or "\\" in raw or relative.is_absolute() or ".." in relative.parts or relative.parts != expected_parts:
        raise ReaderPromoterEvidenceIntegrityError(
            "Promoter artifact path must be a confined content-addressed relative path."
        )
    artifact = (manifest.parent / Path(*relative.parts)).resolve()
    try:
        artifact.relative_to(manifest.parent)
    except ValueError as exc:
        raise ReaderPromoterEvidenceIntegrityError("Promoter artifact path escapes its evidence bundle.") from exc
    return artifact, manifest


def _verify_display_identity(row: Mapping[str, Any]) -> None:
    identity_fields = ("candidate_id", "design_id", "reader_experiment_id", "reduction_id")
    if any(not isinstance(row.get(field), str) or not str(row.get(field)).strip() for field in identity_fields):
        raise ReaderPromoterEvidenceIntegrityError("Promoter display selection identity is incomplete.")
    selected_binding = row.get("selected_binding")
    if (
        row.get("id") != row.get("candidate_id")
        or not isinstance(selected_binding, Mapping)
        or selected_binding.get("reader_design_id") != row.get("design_id")
        or selected_binding.get("candidate_id") != row.get("candidate_id")
        or selected_binding.get("binding_status") != "resolved"
        or selected_binding.get("binding_method") != "exact_alias"
    ):
        raise ReaderPromoterEvidenceIntegrityError("Promoter display identity or selected binding is invalid.")


def _verify_publication_manifest_binding(row: Mapping[str, Any], *, manifest_path: Path) -> None:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReaderPromoterEvidenceIntegrityError(f"Promoter display manifest is not valid JSON: {exc}") from exc
    try:
        projection = parse_reader_evidence_manifest_adapter(payload)
    except ReaderEvidenceManifestAdapterError as exc:
        raise ReaderPromoterEvidenceIntegrityError(
            f"Promoter display manifest Reader adapter is invalid: {exc}"
        ) from exc
    publication_rows = projection.rows
    matches: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for publication_row in publication_rows:
        if not isinstance(publication_row, Mapping):
            continue
        artifacts = publication_row.get("artifacts")
        if not isinstance(artifacts, list):
            continue
        for artifact in artifacts:
            if isinstance(artifact, Mapping) and artifact.get("path") == row.get("path"):
                matches.append((publication_row, artifact))
    if len(matches) != 1:
        raise ReaderPromoterEvidenceIntegrityError(
            "Promoter display artifact must resolve exactly once in its publication manifest."
        )
    publication_row, publication_artifact = matches[0]
    row_fields = {
        "id": row.get("id"),
        "candidate_id": row.get("candidate_id"),
        "design_id": row.get("design_id"),
        "reader_experiment_id": row.get("reader_experiment_id"),
        "reduction_id": row.get("reduction_id"),
        "evidence_role": row.get("evidence_role"),
        "claim_status": row.get("claim_status"),
        "selected_binding": row.get("selected_binding"),
        "binding_source": row.get("binding_source"),
    }
    if any(publication_row.get(field) != value for field, value in row_fields.items()):
        raise ReaderPromoterEvidenceIntegrityError(
            "Promoter display identity or binding disagrees with its publication manifest."
        )
    artifact_fields = {
        "semantic_kind": row.get("semantic_kind"),
        "kind": row.get("kind"),
        "record_id": row.get("artifact_record_id"),
        "scope": row.get("scope"),
        "path": row.get("path"),
        "path_label": row.get("path_label"),
        "exists": row.get("exists"),
        "media_type": row.get("media_type"),
        "bytes": row.get("bytes"),
        "sha256": row.get("sha256"),
        "source_manifest_sha256": row.get("source_manifest_sha256"),
    }
    if any(publication_artifact.get(field) != value for field, value in artifact_fields.items()):
        raise ReaderPromoterEvidenceIntegrityError(
            "Promoter display artifact metadata disagrees with its publication manifest."
        )


def _verify_digest(path: Path, *, expected: object, label: str) -> None:
    if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
        raise ReaderPromoterEvidenceIntegrityError(f"Promoter {label} SHA-256 metadata is invalid.")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = "sha256:" + digest.hexdigest()
    if actual != expected:
        raise ReaderPromoterEvidenceIntegrityError(f"Promoter {label} SHA-256 no longer matches its manifest.")


__all__ = [
    "PROMOTER_RESPONSE_EVIDENCE_SEMANTIC_KIND",
    "READER_PROMOTER_EVIDENCE_MAX_BYTES",
    "ReaderPromoterEvidenceIntegrityError",
    "is_reader_promoter_evidence_artifact",
    "verify_reader_promoter_evidence_artifact",
]
