"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/reader_promoter_evidence/contracts.py

Contracts for study-owned Reader promoter-evidence display manifests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

READER_BUNDLE_SCHEMA_VERSION = "reader.response_window.promoter_evidence_bundle.v4"
READER_EVIDENCE_SCHEMA_VERSION = "stress_ethanol_cipro_growth.reader_promoter_evidence.v1"
# Wire identity declared by the OPAL consumer contract. The study publishes the
# value without importing OPAL, preserving the study-to-campaign boundary.
READER_EVIDENCE_MANIFEST_ADAPTER = "opal.reader_evidence_manifest.v1"
READER_PROMOTER_EVIDENCE_FILENAME = "reader_evidence_promoter_response.json"
READER_PROMOTER_EVIDENCE_MEDIA_DIR = "reader_evidence_media"
TARGET_CAMPAIGN_SLUG = "secg_msrb_greedy"
PROMOTER_RESPONSE_SEMANTIC_KIND = "promoter_response_evidence"
PROMOTER_EVIDENCE_ARTIFACT_IDS = ("promoter_evidence.png", "promoter_evidence.pdf")
PROMOTER_EVIDENCE_NON_CLAIM = (
    "Reader presents response-window evidence and sequence context; downstream objective scoring, "
    "normalization or calibration, and promotion remain outside Reader."
)


class ReaderPromoterEvidenceError(ValueError):
    """Raised when Reader promoter evidence violates the study handoff contract."""


@dataclass(frozen=True)
class VerifiedReaderPromoterEvidenceBundle:
    """One independently verified Reader promoter-evidence bundle."""

    root: Path
    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, Any]


@dataclass(frozen=True)
class ReaderPromoterEvidenceWriteResult:
    """One atomically materialized display-only manifest."""

    manifest_json: Path
    row_count: int
    artifact_count: int


@dataclass(frozen=True)
class ReaderPromoterEvidenceVerification:
    """Verified identity and counts for a display-only manifest."""

    manifest_json: Path
    row_count: int
    artifact_count: int


__all__ = [
    "TARGET_CAMPAIGN_SLUG",
    "PROMOTER_EVIDENCE_ARTIFACT_IDS",
    "PROMOTER_EVIDENCE_NON_CLAIM",
    "PROMOTER_RESPONSE_SEMANTIC_KIND",
    "READER_BUNDLE_SCHEMA_VERSION",
    "READER_EVIDENCE_SCHEMA_VERSION",
    "READER_EVIDENCE_MANIFEST_ADAPTER",
    "READER_PROMOTER_EVIDENCE_FILENAME",
    "READER_PROMOTER_EVIDENCE_MEDIA_DIR",
    "ReaderPromoterEvidenceError",
    "ReaderPromoterEvidenceVerification",
    "ReaderPromoterEvidenceWriteResult",
    "VerifiedReaderPromoterEvidenceBundle",
]
