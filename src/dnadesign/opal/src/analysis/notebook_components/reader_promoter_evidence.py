"""Render-time integrity checks for static Reader promoter evidence."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

PROMOTER_RESPONSE_EVIDENCE_SEMANTIC_KIND = "promoter_response_evidence"
PROMOTER_EVIDENCE_BUNDLE_RECORD_ID = "reader.response_window.promoter_evidence_bundle.v1"
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
    artifact_path = _exact_absolute_path(row.get("path"), label="artifact path")
    source_manifest = _exact_absolute_path(row.get("source_manifest_path"), label="source manifest path")
    if source_manifest.name != "manifest.json" or artifact_path.parent != source_manifest.parent:
        raise ReaderPromoterEvidenceIntegrityError(
            "Promoter artifact and source manifest must share one exact bundle root."
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
    if not source_manifest.is_file():
        raise ReaderPromoterEvidenceIntegrityError("Promoter source manifest is missing.")
    _verify_digest(source_manifest, expected=row.get("source_manifest_sha256"), label="source manifest")
    _verify_source_manifest_binding(row, source_manifest=source_manifest, artifact_path=artifact_path)
    with artifact_path.open("rb") as stream:
        signature = stream.read(8)
    if artifact_path.suffix == ".png" and signature != b"\x89PNG\r\n\x1a\n":
        raise ReaderPromoterEvidenceIntegrityError("Promoter PNG signature is invalid.")
    if artifact_path.suffix == ".pdf" and not signature.startswith(b"%PDF"):
        raise ReaderPromoterEvidenceIntegrityError("Promoter PDF signature is invalid.")
    return artifact_path


def _exact_absolute_path(value: object, *, label: str) -> Path:
    raw = str(value or "")
    path = Path(raw).expanduser()
    if not raw or raw != str(path) or not path.is_absolute() or path.resolve() != path:
        raise ReaderPromoterEvidenceIntegrityError(f"Promoter {label} must be an exact absolute path.")
    return path


def _verify_source_manifest_binding(
    row: Mapping[str, Any],
    *,
    source_manifest: Path,
    artifact_path: Path,
) -> None:
    try:
        payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReaderPromoterEvidenceIntegrityError(f"Promoter source manifest is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != PROMOTER_EVIDENCE_BUNDLE_RECORD_ID:
        raise ReaderPromoterEvidenceIntegrityError("Promoter source manifest schema is invalid.")
    selection = payload.get("selection")
    selected_binding = payload.get("selected_binding")
    source_artifacts = payload.get("artifacts")
    if not isinstance(selection, dict) or set(selection) != {
        "candidate_id",
        "design_id",
        "experiment_id",
        "reduction_id",
    }:
        raise ReaderPromoterEvidenceIntegrityError("Promoter source selection is malformed.")
    expected_selection = (
        selection["candidate_id"],
        selection["design_id"],
        selection["experiment_id"],
        selection["reduction_id"],
    )
    actual_selection = (
        row.get("candidate_id"),
        row.get("design_id"),
        row.get("reader_experiment_id"),
        row.get("reduction_id"),
    )
    if (
        actual_selection != expected_selection
        or row.get("id") != selection["candidate_id"]
        or row.get("claim_status") != payload.get("claim_status")
        or not isinstance(selected_binding, dict)
        or row.get("selected_binding") != selected_binding
        or selected_binding.get("binding_status") != "resolved"
        or selected_binding.get("binding_method") != "exact_alias"
    ):
        raise ReaderPromoterEvidenceIntegrityError(
            "Promoter display identity, claim, or selected binding disagrees with its source selection."
        )
    if not isinstance(source_artifacts, dict):
        raise ReaderPromoterEvidenceIntegrityError("Promoter source artifact records are malformed.")
    source_record = source_artifacts.get(artifact_path.name)
    if not isinstance(source_record, dict) or set(source_record) != {"path", "bytes", "sha256"}:
        raise ReaderPromoterEvidenceIntegrityError("Promoter source artifact record is malformed.")
    if (
        source_record["path"] != artifact_path.name
        or source_record["bytes"] != row.get("bytes")
        or source_record["sha256"] != row.get("sha256")
    ):
        raise ReaderPromoterEvidenceIntegrityError("Promoter display artifact disagrees with its source manifest.")


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
