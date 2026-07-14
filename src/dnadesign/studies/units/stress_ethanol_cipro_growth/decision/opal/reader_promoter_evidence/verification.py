"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/reader_promoter_evidence/verification.py

Independent verification of Reader promoter-evidence bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from .binding_verification import verify_reader_study_binding
from .bundle_semantics import verify_reader_bundle_semantics
from .contracts import (
    PROMOTER_EVIDENCE_ARTIFACT_IDS,
    PROMOTER_EVIDENCE_NON_CLAIM,
    READER_BUNDLE_SCHEMA_VERSION,
    ReaderPromoterEvidenceError,
    VerifiedReaderPromoterEvidenceBundle,
)

_MANIFEST_FIELDS = {
    "schema_version",
    "created_at",
    "claim_status",
    "non_claim_boundary",
    "selection",
    "selected_binding",
    "sources",
    "objective_overlay",
    "artifacts",
}


def verify_reader_promoter_evidence_bundle(
    bundle_dir: Path,
    *,
    bindings_bundle: Path,
) -> VerifiedReaderPromoterEvidenceBundle:
    """Independently verify one Reader bundle without importing Reader."""

    root = Path(bundle_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ReaderPromoterEvidenceError(f"Reader promoter-evidence manifest not found: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReaderPromoterEvidenceError(f"Could not parse Reader promoter-evidence manifest: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != _MANIFEST_FIELDS:
        raise ReaderPromoterEvidenceError(
            f"Reader promoter-evidence manifest fields must be exactly {sorted(_MANIFEST_FIELDS)}."
        )
    if payload["schema_version"] != READER_BUNDLE_SCHEMA_VERSION:
        raise ReaderPromoterEvidenceError(f"Reader promoter evidence must use {READER_BUNDLE_SCHEMA_VERSION!r}.")
    _verify_created_at(payload["created_at"])
    claim_status = payload["claim_status"]
    if claim_status not in {"objective_neutral", "screen_only"}:
        raise ReaderPromoterEvidenceError("Reader promoter evidence has an unsupported claim_status.")
    if payload["non_claim_boundary"] != PROMOTER_EVIDENCE_NON_CLAIM:
        raise ReaderPromoterEvidenceError("Reader promoter evidence changed its non-claim boundary.")
    verify_reader_bundle_semantics(payload, claim_status=str(claim_status))
    verify_reader_study_binding(payload, bindings_bundle=bindings_bundle)
    artifacts = payload["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != set(PROMOTER_EVIDENCE_ARTIFACT_IDS):
        raise ReaderPromoterEvidenceError(
            f"Reader promoter-evidence artifacts must be exactly {sorted(PROMOTER_EVIDENCE_ARTIFACT_IDS)}."
        )
    for artifact_id in PROMOTER_EVIDENCE_ARTIFACT_IDS:
        _verify_artifact(root, artifact_id=artifact_id, value=artifacts[artifact_id])
    return VerifiedReaderPromoterEvidenceBundle(
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=_sha256(manifest_path),
        manifest=payload,
    )


def _verify_artifact(root: Path, *, artifact_id: str, value: object) -> None:
    if not isinstance(value, dict) or set(value) != {"path", "bytes", "sha256"}:
        raise ReaderPromoterEvidenceError(f"Reader artifact {artifact_id!r} metadata is malformed.")
    if value["path"] != artifact_id:
        raise ReaderPromoterEvidenceError(f"Reader artifact {artifact_id!r} path disagrees with its identity.")
    path = (root / artifact_id).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ReaderPromoterEvidenceError(f"Reader artifact {artifact_id!r} escapes its bundle root.") from exc
    size = value["bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 1 or not path.is_file():
        raise ReaderPromoterEvidenceError(f"Reader artifact {artifact_id!r} is missing or has an invalid size.")
    if path.stat().st_size != size or _sha256(path) != value["sha256"]:
        raise ReaderPromoterEvidenceError(f"Reader artifact {artifact_id!r} digest or size mismatch.")
    signature = path.read_bytes()[:8]
    if artifact_id.endswith(".pdf") and not signature.startswith(b"%PDF"):
        raise ReaderPromoterEvidenceError("Reader promoter-evidence PDF signature is invalid.")
    if artifact_id.endswith(".png") and signature != b"\x89PNG\r\n\x1a\n":
        raise ReaderPromoterEvidenceError("Reader promoter-evidence PNG signature is invalid.")


def _verify_created_at(value: object) -> None:
    if not isinstance(value, str):
        raise ReaderPromoterEvidenceError("Reader created_at must be an ISO-8601 timestamp.")
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ReaderPromoterEvidenceError("Reader created_at must be an ISO-8601 timestamp.") from exc
    if timestamp.tzinfo is None:
        raise ReaderPromoterEvidenceError("Reader created_at must include a timezone.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


__all__ = ["verify_reader_promoter_evidence_bundle"]
