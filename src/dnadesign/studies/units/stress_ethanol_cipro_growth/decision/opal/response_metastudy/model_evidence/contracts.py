"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/model_evidence/contracts.py

Contracts for immutable model-evidence trajectory records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

PROTOCOL_ID = "stress_response_window_grouped_model_evidence_v2"
PROTOCOL_SCHEMA_VERSION = "stress_ethanol_cipro_growth.model_evidence_protocol.v2"
CHECKPOINT_SCHEMA_VERSION = "stress_ethanol_cipro_growth.model_evidence_checkpoint.v2"
LATEST_SCHEMA_VERSION = "stress_ethanol_cipro_growth.model_evidence_latest.v1"
CATALOG_SCHEMA_VERSION = "stress_ethanol_cipro_growth.model_evidence_catalog.v1"

_EVIDENCE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class ModelEvidenceError(RuntimeError):
    """Raised when model-evidence trajectory integrity cannot be established."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ModelEvidenceError(f"model-evidence record is not canonical JSON: {exc}") from exc
    return rendered.encode("utf-8")


def content_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def validated_evidence_id(value: object) -> str:
    evidence_id = str(value)
    if _EVIDENCE_ID.fullmatch(evidence_id) is None:
        raise ModelEvidenceError(
            "evidence_id must be 1-128 characters using letters, numbers, period, underscore, or hyphen."
        )
    return evidence_id


__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "CHECKPOINT_SCHEMA_VERSION",
    "LATEST_SCHEMA_VERSION",
    "PROTOCOL_ID",
    "PROTOCOL_SCHEMA_VERSION",
    "ModelEvidenceError",
    "canonical_json_bytes",
    "content_digest",
    "validated_evidence_id",
]
