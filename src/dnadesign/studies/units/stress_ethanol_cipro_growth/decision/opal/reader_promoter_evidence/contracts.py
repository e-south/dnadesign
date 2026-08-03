"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/reader_promoter_evidence/contracts.py

Contracts for the study-owned Reader diagnostic display projection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.reader_records import (
    ReaderResponseDisplay,
    ReaderResponseRecords,
)

READER_EVENT_WINDOW_DIAGNOSTIC_RECORD_ID = "plot:four_state_event_window_diagnostic"
READER_EVIDENCE_SCHEMA_VERSION = "stress_ethanol_cipro_growth.reader_promoter_evidence.v3"
# Wire identity declared by OPAL's producer-neutral consumer contract.
READER_EVIDENCE_MANIFEST_ADAPTER = "opal.reader_evidence_manifest.v1"
READER_PROMOTER_EVIDENCE_FILENAME = "reader_evidence_promoter_response.json"
READER_PROMOTER_EVIDENCE_MEDIA_DIR = "reader_evidence_media"
TARGET_CAMPAIGN_SLUG = "secg_msrb_greedy"
PROMOTER_RESPONSE_SEMANTIC_KIND = "promoter_response_evidence"
PROMOTER_EVIDENCE_NON_CLAIM = (
    "Reader publishes verified response-window records and diagnostic media; the stress study binds "
    "candidate identity and display meaning. Objective scoring, label promotion, and campaign state are separate."
)


class ReaderPromoterEvidenceError(ValueError):
    """Raised when the promoter display projection fails closed."""


@dataclass(frozen=True)
class VerifiedReaderPromoterEvidenceSource:
    """One canonical Reader diagnostic joined to one exact study candidate."""

    records: ReaderResponseRecords
    display: ReaderResponseDisplay
    selected_binding: dict[str, object]
    binding_source: dict[str, str]

    @property
    def candidate_id(self) -> str:
        return str(self.selected_binding["candidate_id"])

    @property
    def design_id(self) -> str:
        return self.display.design_id

    @property
    def source_experiment_id(self) -> str:
        return self.display.source_experiment_id

    @property
    def reduction_id(self) -> str:
        return self.records.primary_reduction_id


@dataclass(frozen=True)
class ReaderPromoterEvidenceWriteResult:
    """One atomically materialized display-only manifest."""

    manifest_json: Path
    row_count: int
    artifact_count: int


@dataclass(frozen=True)
class ReaderPromoterEvidenceVerification:
    """Verified identity and counts for one display-only manifest."""

    manifest_json: Path
    row_count: int
    artifact_count: int


def canonical_json_sha256(payload: Any) -> str:
    """Return the canonical digest used to bind one source receipt."""

    import hashlib
    import json

    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


__all__ = [
    "PROMOTER_EVIDENCE_NON_CLAIM",
    "PROMOTER_RESPONSE_SEMANTIC_KIND",
    "READER_EVENT_WINDOW_DIAGNOSTIC_RECORD_ID",
    "READER_EVIDENCE_MANIFEST_ADAPTER",
    "READER_EVIDENCE_SCHEMA_VERSION",
    "READER_PROMOTER_EVIDENCE_FILENAME",
    "READER_PROMOTER_EVIDENCE_MEDIA_DIR",
    "TARGET_CAMPAIGN_SLUG",
    "ReaderPromoterEvidenceError",
    "ReaderPromoterEvidenceVerification",
    "ReaderPromoterEvidenceWriteResult",
    "VerifiedReaderPromoterEvidenceSource",
    "canonical_json_sha256",
]
