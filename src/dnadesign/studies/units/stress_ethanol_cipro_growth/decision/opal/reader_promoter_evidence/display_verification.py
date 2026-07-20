"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/reader_promoter_evidence/display_verification.py

Verification for study-owned Reader evidence display manifests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .bundle_semantics import verify_reader_bundle_semantics
from .contracts import (
    PROMOTER_EVIDENCE_ARTIFACT_IDS,
    PROMOTER_EVIDENCE_NON_CLAIM,
    READER_EVIDENCE_MANIFEST_ADAPTER,
    READER_EVIDENCE_SCHEMA_VERSION,
    TARGET_CAMPAIGN_SLUG,
    ReaderPromoterEvidenceError,
    ReaderPromoterEvidenceVerification,
)
from .display_artifact_verification import is_sha256, verify_display_artifact

_DISPLAY_MANIFEST_FIELDS = {
    "schema_version",
    "opal_adapter",
    "created_at",
    "campaign_slug",
    "round",
    "summary",
    "rows",
}
_DISPLAY_SUMMARY_FIELDS = {"rows", "distinct_ids", "reader_experiments", "artifact_count", "missing_artifact_rows"}
_DISPLAY_ROW_FIELDS = {
    "id",
    "candidate_id",
    "design_id",
    "reader_experiment_id",
    "reduction_id",
    "evidence_role",
    "claim_status",
    "non_claim_boundary",
    "selected_binding",
    "sources",
    "objective_overlay",
    "artifacts",
}


def verify_reader_promoter_evidence_manifest(path: Path) -> ReaderPromoterEvidenceVerification:
    """Verify a display manifest and every exact Reader source binding."""

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise ReaderPromoterEvidenceError(f"Reader promoter-evidence display manifest not found: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ReaderPromoterEvidenceError(f"Could not parse Reader display manifest: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != _DISPLAY_MANIFEST_FIELDS:
        raise ReaderPromoterEvidenceError(
            f"Reader display manifest fields must be exactly {sorted(_DISPLAY_MANIFEST_FIELDS)}."
        )
    if payload["schema_version"] != READER_EVIDENCE_SCHEMA_VERSION:
        raise ReaderPromoterEvidenceError(f"Reader display manifest must use {READER_EVIDENCE_SCHEMA_VERSION!r}.")
    if payload["opal_adapter"] != READER_EVIDENCE_MANIFEST_ADAPTER:
        raise ReaderPromoterEvidenceError(f"Reader display opal_adapter must use {READER_EVIDENCE_MANIFEST_ADAPTER!r}.")
    _display_created_at(payload["created_at"])
    if payload["campaign_slug"] != TARGET_CAMPAIGN_SLUG:
        raise ReaderPromoterEvidenceError(f"Reader display campaign_slug must be {TARGET_CAMPAIGN_SLUG!r}.")
    round_label = payload["round"]
    if not isinstance(round_label, str) or not round_label.startswith("r") or not round_label[1:].isdigit():
        raise ReaderPromoterEvidenceError("Reader display round must use the OPAL form 'r<integer>'.")
    rows = payload["rows"]
    if not isinstance(rows, list) or not rows:
        raise ReaderPromoterEvidenceError("Reader display manifest rows must be a non-empty list.")
    verified_rows = [_verify_display_row(row, manifest_root=manifest_path.parent) for row in rows]
    identities = [row["identity"] for row in verified_rows]
    if len(identities) != len(set(identities)):
        raise ReaderPromoterEvidenceError("Reader display manifest contains duplicate selection identities.")
    summary = payload["summary"]
    if not isinstance(summary, dict) or set(summary) != _DISPLAY_SUMMARY_FIELDS:
        raise ReaderPromoterEvidenceError(
            f"Reader display summary fields must be exactly {sorted(_DISPLAY_SUMMARY_FIELDS)}."
        )
    expected_summary = {
        "rows": len(rows),
        "distinct_ids": len({row["candidate_id"] for row in verified_rows}),
        "reader_experiments": len({row["experiment_id"] for row in verified_rows}),
        "artifact_count": sum(row["artifact_count"] for row in verified_rows),
        "missing_artifact_rows": 0,
    }
    if summary != expected_summary:
        raise ReaderPromoterEvidenceError(
            f"Reader display summary mismatch: expected={expected_summary} actual={summary}"
        )
    return ReaderPromoterEvidenceVerification(
        manifest_json=manifest_path,
        row_count=len(rows),
        artifact_count=expected_summary["artifact_count"],
    )


def _verify_display_row(value: object, *, manifest_root: Path) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _DISPLAY_ROW_FIELDS:
        raise ReaderPromoterEvidenceError(f"Reader display row fields must be exactly {sorted(_DISPLAY_ROW_FIELDS)}.")
    text_fields = ("id", "candidate_id", "design_id", "reader_experiment_id", "reduction_id")
    if any(not _nonempty(value[field]) for field in text_fields):
        raise ReaderPromoterEvidenceError("Reader display identity values must be non-empty strings.")
    if value["id"] != value["candidate_id"]:
        raise ReaderPromoterEvidenceError("Reader display id must equal candidate_id.")
    if value["evidence_role"] != "display_only":
        raise ReaderPromoterEvidenceError("Reader promoter evidence must remain display_only.")
    if value["claim_status"] not in {"objective_neutral", "screen_only"}:
        raise ReaderPromoterEvidenceError("Reader display claim_status is unsupported.")
    if value["non_claim_boundary"] != PROMOTER_EVIDENCE_NON_CLAIM:
        raise ReaderPromoterEvidenceError("Reader display changed the promoter-evidence non-claim boundary.")
    verify_reader_bundle_semantics(
        {
            "selection": {
                "experiment_id": str(value["reader_experiment_id"]),
                "design_id": str(value["design_id"]),
                "candidate_id": str(value["candidate_id"]),
                "reduction_id": str(value["reduction_id"]),
            },
            "sources": value["sources"],
            "selected_binding": value["selected_binding"],
            "objective_overlay": value["objective_overlay"],
        },
        claim_status=str(value["claim_status"]),
    )
    artifacts = value["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != len(PROMOTER_EVIDENCE_ARTIFACT_IDS):
        raise ReaderPromoterEvidenceError("Reader display row must contain exactly its PNG and PDF artifacts.")
    actual_identity = (
        str(value["candidate_id"]),
        str(value["design_id"]),
        str(value["reader_experiment_id"]),
        str(value["reduction_id"]),
    )
    source_digests = {artifact.get("source_manifest_sha256") for artifact in artifacts if isinstance(artifact, dict)}
    if len(source_digests) != 1 or not is_sha256(next(iter(source_digests))):
        raise ReaderPromoterEvidenceError("Reader display artifacts must share one Reader source-manifest digest.")
    source_manifest_sha256 = str(next(iter(source_digests)))
    artifact_ids = [
        verify_display_artifact(
            artifact,
            manifest_root=manifest_root,
            identity=actual_identity,
            source_manifest_sha256=source_manifest_sha256,
        )
        for artifact in artifacts
    ]
    if set(artifact_ids) != set(PROMOTER_EVIDENCE_ARTIFACT_IDS):
        raise ReaderPromoterEvidenceError("Reader display row does not bind the exact PNG and PDF artifacts.")
    return {
        "identity": actual_identity,
        "candidate_id": actual_identity[0],
        "experiment_id": actual_identity[2],
        "artifact_count": len(artifacts),
    }


def _display_created_at(value: object) -> None:
    if not isinstance(value, str):
        raise ReaderPromoterEvidenceError("Reader display created_at must be an ISO-8601 timestamp.")
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ReaderPromoterEvidenceError("Reader display created_at must be an ISO-8601 timestamp.") from exc
    if timestamp.tzinfo is None:
        raise ReaderPromoterEvidenceError("Reader display created_at must include a timezone.")


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


__all__ = ["verify_reader_promoter_evidence_manifest"]
