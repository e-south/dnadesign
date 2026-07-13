"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/promoter_candidate_bindings/manifest_contract.py

Manifest contract for study-owned promoter candidate bindings.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Any

import pandas as pd

from dnadesign.baserender import (
    BASERENDER_SEQUENCE_PANEL_CONTRACT_ID,
    BASERENDER_SEQUENCE_PANEL_CONTRACT_VERSION,
)

from .contracts import (
    BINDINGS_FILENAME,
    BINDINGS_RECORD_ID,
    SCHEMA_ID,
    SCHEMA_VERSION,
    STUDY_ID,
    BindingSourceArtifact,
    PromoterCandidateBindingsError,
    PromoterCandidateBindingsPreview,
)
from .values import required_sha256, required_text

MANIFEST_FIELDS = {
    "schema_id",
    "schema_version",
    "study_id",
    "created_at",
    "record",
    "candidate_table",
    "baserender_contract",
    "source_artifacts",
}
RECORD_FIELDS = {"record_id", "path", "sha256", "row_count"}


def build_manifest(
    preview: PromoterCandidateBindingsPreview,
    *,
    bindings_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "created_at": datetime.now(UTC).isoformat(),
        "record": {
            "record_id": BINDINGS_RECORD_ID,
            "path": BINDINGS_FILENAME,
            "sha256": bindings_sha256,
            "row_count": int(len(preview.bindings)),
        },
        "candidate_table": {
            "dataset_id": preview.candidate_table_id,
            "selection_sha256": preview.candidate_selection_sha256,
        },
        "baserender_contract": {
            "contract_id": BASERENDER_SEQUENCE_PANEL_CONTRACT_ID,
            "contract_version": BASERENDER_SEQUENCE_PANEL_CONTRACT_VERSION,
        },
        "source_artifacts": [
            {"artifact_id": item.artifact_id, "path": item.path, "sha256": item.sha256}
            for item in sorted(preview.source_artifacts, key=lambda source: source.artifact_id)
        ],
    }


def validate_manifest(payload: dict[str, Any], *, bindings: pd.DataFrame) -> None:
    if set(payload) != MANIFEST_FIELDS:
        raise PromoterCandidateBindingsError("Promoter binding manifest fields do not match the v1 contract.")
    if (
        payload.get("schema_id") != SCHEMA_ID
        or str(payload.get("schema_version")) != SCHEMA_VERSION
        or payload.get("study_id") != STUDY_ID
    ):
        raise PromoterCandidateBindingsError("Promoter binding manifest identity does not match this study contract.")
    _aware_timestamp(payload.get("created_at"))
    _validate_record(payload.get("record"), bindings=bindings)
    _validate_candidate_table(payload.get("candidate_table"), bindings=bindings)
    expected_baserender = {
        "contract_id": BASERENDER_SEQUENCE_PANEL_CONTRACT_ID,
        "contract_version": BASERENDER_SEQUENCE_PANEL_CONTRACT_VERSION,
    }
    if payload.get("baserender_contract") != expected_baserender:
        raise PromoterCandidateBindingsError("Promoter binding BaseRender contract mismatch.")
    validate_source_artifacts(parse_source_artifacts(payload.get("source_artifacts")))


def parse_source_artifacts(value: object) -> tuple[BindingSourceArtifact, ...]:
    if not isinstance(value, list) or not value:
        raise PromoterCandidateBindingsError("Promoter binding source_artifacts must be non-empty.")
    parsed: list[BindingSourceArtifact] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"artifact_id", "path", "sha256"}:
            raise PromoterCandidateBindingsError("Promoter binding source-artifact entry is malformed.")
        parsed.append(
            BindingSourceArtifact(
                artifact_id=required_text(item["artifact_id"], field="source artifact ID"),
                path=relative_path(item["path"], field="source artifact path"),
                sha256=required_sha256(item["sha256"], field="source artifact SHA-256"),
            )
        )
    return tuple(parsed)


def validate_source_artifacts(sources: Sequence[BindingSourceArtifact]) -> None:
    if not sources:
        raise PromoterCandidateBindingsError("Promoter binding sources must be non-empty.")
    ids = [source.artifact_id for source in sources]
    if len(ids) != len(set(ids)):
        raise PromoterCandidateBindingsError("Promoter binding source artifact IDs must be unique.")
    for source in sources:
        required_text(source.artifact_id, field="source artifact ID")
        if relative_path(source.path, field="source artifact path") != source.path:
            raise PromoterCandidateBindingsError("Source artifact paths must be normalized POSIX paths.")
        required_sha256(source.sha256, field="source artifact SHA-256")


def relative_path(value: object, *, field: str) -> str:
    text = required_text(value, field=field)
    path = PurePosixPath(text)
    first = path.parts[0] if path.parts else ""
    if "\\" in text or text.startswith("~") or path.is_absolute() or ".." in path.parts or ":" in first:
        raise PromoterCandidateBindingsError(f"{field} must be a confined relative POSIX path.")
    return str(path)


def _validate_record(value: object, *, bindings: pd.DataFrame) -> None:
    if not isinstance(value, dict) or set(value) != RECORD_FIELDS:
        raise PromoterCandidateBindingsError("Promoter binding record is malformed.")
    if value.get("record_id") != BINDINGS_RECORD_ID or value.get("path") != BINDINGS_FILENAME:
        raise PromoterCandidateBindingsError("Promoter binding record identity mismatch.")
    row_count = value.get("row_count")
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count <= 0:
        raise PromoterCandidateBindingsError("Promoter binding record row_count must be a positive integer.")
    if row_count != len(bindings):
        raise PromoterCandidateBindingsError("Promoter binding record row-count mismatch.")
    required_sha256(value.get("sha256"), field="binding record SHA-256")


def _validate_candidate_table(value: object, *, bindings: pd.DataFrame) -> None:
    if not isinstance(value, dict) or set(value) != {"dataset_id", "selection_sha256"}:
        raise PromoterCandidateBindingsError("Promoter binding candidate-table provenance is malformed.")
    dataset_id = required_text(value["dataset_id"], field="candidate table ID")
    digest = required_sha256(value["selection_sha256"], field="candidate selection SHA-256")
    if set(bindings["candidate_table_id"].astype(str)) != {dataset_id} or set(
        bindings["candidate_selection_sha256"].astype(str)
    ) != {digest}:
        raise PromoterCandidateBindingsError("Binding rows disagree with candidate-table provenance.")


def _aware_timestamp(value: object) -> datetime:
    text = required_text(value, field="manifest created_at")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PromoterCandidateBindingsError("Manifest created_at must be an ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PromoterCandidateBindingsError("Manifest created_at must include a UTC offset.")
    return parsed.astimezone(UTC)
