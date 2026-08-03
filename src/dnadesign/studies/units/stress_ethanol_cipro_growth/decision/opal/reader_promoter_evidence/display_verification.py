"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/reader_promoter_evidence/display_verification.py

Verify the study-owned Reader diagnostic display manifest.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .contracts import (
    PROMOTER_EVIDENCE_NON_CLAIM,
    READER_EVIDENCE_MANIFEST_ADAPTER,
    READER_EVIDENCE_SCHEMA_VERSION,
    TARGET_CAMPAIGN_SLUG,
    ReaderPromoterEvidenceError,
    ReaderPromoterEvidenceVerification,
)
from .display_artifact_verification import verify_display_artifact
from .source_verification import verify_display_sources, verify_selected_binding

_MANIFEST_FIELDS = {"schema_version", "opal_adapter", "created_at", "campaign_slug", "round", "summary", "rows"}
_SUMMARY_FIELDS = {"rows", "distinct_ids", "reader_experiments", "artifact_count", "missing_artifact_rows"}
_ROW_FIELDS = {
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
    "artifacts",
}


def verify_reader_promoter_evidence_manifest(path: Path) -> ReaderPromoterEvidenceVerification:
    """Verify one portable display manifest without reading its original Reader checkout."""

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise ReaderPromoterEvidenceError(f"Reader promoter-evidence display manifest not found: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReaderPromoterEvidenceError(f"Could not parse Reader display manifest: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != _MANIFEST_FIELDS:
        raise ReaderPromoterEvidenceError(f"Reader display manifest fields must be exactly {sorted(_MANIFEST_FIELDS)}.")
    if payload["schema_version"] != READER_EVIDENCE_SCHEMA_VERSION:
        raise ReaderPromoterEvidenceError(f"Reader display manifest must use {READER_EVIDENCE_SCHEMA_VERSION!r}.")
    if payload["opal_adapter"] != READER_EVIDENCE_MANIFEST_ADAPTER:
        raise ReaderPromoterEvidenceError(f"Reader display opal_adapter must use {READER_EVIDENCE_MANIFEST_ADAPTER!r}.")
    _timestamp(payload["created_at"])
    if payload["campaign_slug"] != TARGET_CAMPAIGN_SLUG:
        raise ReaderPromoterEvidenceError(f"Reader display campaign_slug must be {TARGET_CAMPAIGN_SLUG!r}.")
    round_label = payload["round"]
    if not isinstance(round_label, str) or not round_label.startswith("r") or not round_label[1:].isdigit():
        raise ReaderPromoterEvidenceError("Reader display round must use the OPAL form 'r<integer>'.")
    rows = payload["rows"]
    if not isinstance(rows, list) or not rows:
        raise ReaderPromoterEvidenceError("Reader display manifest rows must be a non-empty list.")
    verified = [_verify_row(row, manifest_root=manifest_path.parent) for row in rows]
    identities = [item["identity"] for item in verified]
    if len(identities) != len(set(identities)):
        raise ReaderPromoterEvidenceError("Reader display manifest contains duplicate selection identities.")
    expected_summary = {
        "rows": len(rows),
        "distinct_ids": len({item["candidate_id"] for item in verified}),
        "reader_experiments": len({item["experiment_id"] for item in verified}),
        "artifact_count": len(rows),
        "missing_artifact_rows": 0,
    }
    if not isinstance(payload["summary"], dict) or set(payload["summary"]) != _SUMMARY_FIELDS:
        raise ReaderPromoterEvidenceError(f"Reader display summary fields must be exactly {sorted(_SUMMARY_FIELDS)}.")
    if payload["summary"] != expected_summary:
        raise ReaderPromoterEvidenceError(
            f"Reader display summary mismatch: expected={expected_summary} actual={payload['summary']}"
        )
    return ReaderPromoterEvidenceVerification(
        manifest_json=manifest_path,
        row_count=len(rows),
        artifact_count=len(rows),
    )


def _verify_row(value: object, *, manifest_root: Path) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _ROW_FIELDS:
        raise ReaderPromoterEvidenceError(f"Reader display row fields must be exactly {sorted(_ROW_FIELDS)}.")
    for field in ("id", "candidate_id", "design_id", "reader_experiment_id", "reduction_id"):
        _text(value[field], field=field)
    if value["id"] != value["candidate_id"]:
        raise ReaderPromoterEvidenceError("Reader display id must equal candidate_id.")
    if value["evidence_role"] != "display_only" or value["claim_status"] != "objective_neutral":
        raise ReaderPromoterEvidenceError(
            "Reader promoter evidence must remain objective-neutral display-only evidence."
        )
    if value["non_claim_boundary"] != PROMOTER_EVIDENCE_NON_CLAIM:
        raise ReaderPromoterEvidenceError("Reader display changed the promoter-evidence non-claim boundary.")
    selected = verify_selected_binding(value["selected_binding"], row=value)
    response = verify_display_sources(value["sources"], row=value)
    artifacts = value["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise ReaderPromoterEvidenceError("Reader display row must contain exactly one pinned diagnostic artifact.")
    identity = (
        str(value["candidate_id"]),
        str(value["design_id"]),
        str(value["reader_experiment_id"]),
        str(value["reduction_id"]),
    )
    verify_display_artifact(
        artifacts[0],
        manifest_root=manifest_root,
        identity=identity,
        response_source=response,
    )
    return {"identity": identity, "candidate_id": selected["candidate_id"], "experiment_id": identity[2]}


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ReaderPromoterEvidenceError(f"Reader display {field} must be trimmed non-empty text.")
    return value


def _timestamp(value: object) -> None:
    if not isinstance(value, str):
        raise ReaderPromoterEvidenceError("Reader display created_at must be an ISO-8601 timestamp.")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ReaderPromoterEvidenceError("Reader display created_at must be an ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None:
        raise ReaderPromoterEvidenceError("Reader display created_at must include a timezone.")


__all__ = ["verify_reader_promoter_evidence_manifest"]
