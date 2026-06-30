"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/manifest.py

Manifest helpers for Eco1 review deliverables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.constants import (
    SCHEMA_ID,
)


def make_deliverable_row(
    *,
    deliverable_id: str,
    section: str,
    artifact_kind: str,
    status: str,
    path: Path,
    source_tables: list[str],
    input_hashes: dict[str, str],
    alt_text: str,
    description: str,
    interpretation_limit: str,
    title: str | None = None,
    method_summary: str = "",
    evidence_summary: dict[str, Any] | None = None,
    role: str = "manuscript_facing",
    render_mode: str = "standard_visual",
    skip_reason: str = "",
) -> dict[str, Any]:
    """Build one manifest row with the required evidence fields."""

    return {
        "deliverable_id": deliverable_id,
        "title": title or _default_title(deliverable_id),
        "section": section,
        "artifact_kind": artifact_kind,
        "status": status,
        "role": role,
        "render_mode": render_mode,
        "path": str(path),
        "source_tables": list(source_tables),
        "input_hashes": dict(input_hashes),
        "alt_text": alt_text,
        "description": description,
        "interpretation_limit": interpretation_limit,
        "method_summary": method_summary,
        "evidence_summary": dict(evidence_summary or {}),
        "skip_reason": skip_reason,
    }


def write_manifest(
    path: Path,
    *,
    deliverables: list[dict[str, Any]],
    notebook_path: Path,
) -> None:
    """Write the top-level review-deliverable manifest."""

    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_root = path.parent
    relative_deliverables = [_with_manifest_relative_path(row, manifest_root) for row in deliverables]
    manifest = {
        "schema_id": SCHEMA_ID,
        "schema_version": 1,
        "status": _manifest_status(relative_deliverables),
        "path_policy": "manifest_relative",
        "deliverable_count": len(relative_deliverables),
        "deliverables": relative_deliverables,
        "notebook": {
            "path": _manifest_relative_path(notebook_path, manifest_root),
            "entrypoint": "marimo run eco1_review_deliverables.py",
            "input_manifest": path.name,
            "scope": "eco1_rt_repack review deliverables",
            "description": (
                "Manifest-backed marimo surface for mask-constraint evidence, "
                "ProteinMPNN variant and fold-triage evidence, and Biohub ESMC "
                "SAE feature-review deliverables."
            ),
        },
        "visual_policy": {
            "requires_alt_text": True,
            "requires_interpretation_limit": True,
            "complete_status": "materialized_complete",
            "degraded_status": "materialized_degraded",
            "degraded_when_statuses": ["skipped_missing_input", "errored"],
            "candidate_acceptance_gate": False,
            "plain_language_limit": (
                "Visuals support review and concise methods writing. They do "
                "not measure RT activity, processivity, strand displacement, "
                "or hairpin readthrough."
            ),
        },
    }
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def _manifest_status(deliverables: list[dict[str, Any]]) -> str:
    degraded_prefixes = ("errored", "skipped_missing_input")
    for row in deliverables:
        status = str(row.get("status") or "")
        if status.startswith(degraded_prefixes):
            return "materialized_degraded"
    return "materialized_complete"


def _with_manifest_relative_path(row: dict[str, Any], manifest_root: Path) -> dict[str, Any]:
    normalized = dict(row)
    normalized["path"] = _manifest_relative_path(Path(str(row["path"])), manifest_root)
    return normalized


def _manifest_relative_path(path: Path, manifest_root: Path) -> str:
    if not path.is_absolute():
        return str(path)
    return os.path.relpath(path, start=manifest_root)


def file_hashes(paths: dict[str, Path]) -> dict[str, str]:
    """Return sha256-prefixed file hashes for existing paths."""

    return {label: "sha256:" + sha256(path) for label, path in paths.items() if path.exists()}


def sha256(path: Path) -> str:
    """Return a SHA-256 digest for a local file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_title(deliverable_id: str) -> str:
    return deliverable_id.replace("_", " ").title()
