"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/reader_evidence_discovery.py

Discovers and projects round-local Reader evidence manifests for notebook use.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dnadesign.opal.api.reader_evidence import (
    ReaderEvidenceManifestAdapterError,
    parse_reader_evidence_manifest_adapter,
)

from ._support import compact_path, mapping, sequence
from .reader_evidence_media import (
    reader_experiment_display_label,
    reader_reduction_display_label,
    reader_round_display_label,
    semantic_kind_label,
    time_selected_label,
)


def discover_reader_evidence_manifests(workdir: str | Path) -> list[dict[str, Any]]:
    """Return small inventory rows for round-local Reader evidence manifests."""

    root = Path(workdir)
    return [
        _reader_evidence_manifest_row(path, workdir=root)
        for path in sorted(root.glob("inputs/r*/reader_evidence*.json"))
    ]


def discover_reader_evidence_artifacts(workdir: str | Path) -> list[dict[str, Any]]:
    """Return flattened artifact rows from round-local Reader evidence manifests."""

    root = Path(workdir)
    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(root.glob("inputs/r*/reader_evidence*.json")):
        payload = _read_payload(manifest_path)
        if payload is None:
            continue
        try:
            projection = parse_reader_evidence_manifest_adapter(payload)
        except ReaderEvidenceManifestAdapterError:
            continue
        for evidence_row in projection.rows:
            item = mapping(evidence_row)
            for artifact in sequence(item.get("artifacts")):
                artifact_item = mapping(artifact)
                path = artifact_item.get("path")
                semantic_kind = str(artifact_item.get("semantic_kind") or "artifact")
                design_id = str(item.get("design_id") or "")
                reader_experiment_id = str(item.get("reader_experiment_id") or "")
                round_label = projection.round_label
                artifact_label = " · ".join(
                    part
                    for part in (
                        reader_round_display_label(round_label),
                        reader_experiment_display_label(reader_experiment_id),
                        design_id,
                        reader_reduction_display_label(item.get("reduction_id")),
                        time_selected_label(item.get("time_selected_h")),
                    )
                    if part
                )
                row = {
                    "label": artifact_label,
                    "round": round_label,
                    "id": str(item.get("id") or ""),
                    "candidate_id": str(item.get("candidate_id") or item.get("id") or ""),
                    "design_id": design_id,
                    "reader_experiment_id": reader_experiment_id,
                    "reduction_id": str(item.get("reduction_id") or ""),
                    "evidence_role": str(item.get("evidence_role") or ""),
                    "claim_status": str(item.get("claim_status") or ""),
                    "non_claim_boundary": str(item.get("non_claim_boundary") or ""),
                    "selected_binding": dict(mapping(item.get("selected_binding"))),
                    "sources": dict(mapping(item.get("sources"))),
                    "objective_overlay": (
                        None if item.get("objective_overlay") is None else dict(mapping(item.get("objective_overlay")))
                    ),
                    "reader_config_path": str(item.get("reader_config_path") or ""),
                    "reader_record_id": str(item.get("reader_record_id") or ""),
                    "sequence": item.get("sequence") or "",
                    "synthesis_name": item.get("synthesis_name") or "",
                    "semantic_kind": semantic_kind,
                    "plot_type_label": semantic_kind_label(semantic_kind),
                    "kind": str(artifact_item.get("kind") or ""),
                    "artifact_record_id": str(artifact_item.get("record_id") or ""),
                    "scope": str(artifact_item.get("scope") or ""),
                    "path": str(path or ""),
                    "path_label": str(artifact_item.get("path_label") or compact_path(path, max_parts=5)),
                    "exists": bool(artifact_item.get("exists")),
                    "media_type": str(artifact_item.get("media_type") or ""),
                    "bytes": artifact_item.get("bytes"),
                    "sha256": str(artifact_item.get("sha256") or ""),
                    "source_manifest_sha256": str(artifact_item.get("source_manifest_sha256") or ""),
                    "manifest_path": str(manifest_path.resolve()),
                    "manifest_path_label": compact_path(manifest_path, base=root),
                }
                if "time_selected_h" in item:
                    row["time_selected_h"] = item.get("time_selected_h")
                if "source_manifest_path" in artifact_item:
                    row["source_manifest_path"] = str(artifact_item.get("source_manifest_path") or "")
                rows.append(row)
    return rows


def _reader_evidence_manifest_row(path: Path, *, workdir: Path) -> dict[str, Any]:
    round_label = _round_label(path)
    row: dict[str, Any] = {
        "path": str(path),
        "path_label": compact_path(path, base=workdir),
        "round": round_label,
        "status": "ready",
        "rows": 0,
        "distinct_ids": 0,
        "reader_experiments": 0,
        "artifact_count": 0,
        "missing_artifact_rows": 0,
    }
    payload = _read_payload(path)
    if payload is None:
        return {**row, "status": "read_error"}
    try:
        projection = parse_reader_evidence_manifest_adapter(payload)
    except ReaderEvidenceManifestAdapterError:
        return {**row, "status": "schema_attention"}
    summary = projection.summary
    row.update(
        {
            "round": projection.round_label,
            "rows": int(summary.get("rows") or 0),
            "distinct_ids": int(summary.get("distinct_ids") or 0),
            "reader_experiments": int(summary.get("reader_experiments") or 0),
            "artifact_count": int(summary.get("artifact_count") or 0),
            "missing_artifact_rows": int(summary.get("missing_artifact_rows") or 0),
        }
    )
    if row["rows"] == 0:
        row["status"] = "empty"
    return row


def _read_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _round_label(path: Path) -> str:
    return path.parent.name if path.parent.name.startswith("r") else ""


__all__ = ["discover_reader_evidence_artifacts", "discover_reader_evidence_manifests"]
