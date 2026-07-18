"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/reader_evidence.py

Builds notebook rows for round-local Reader evidence manifests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from dnadesign.opal.api.reader_evidence import (
    ReaderEvidenceManifestAdapterError,
    parse_reader_evidence_manifest_adapter,
)

from ._support import compact_path, mapping, sequence
from .reader_evidence_media import (
    dedupe_reader_media_labels,
    filter_reader_media_rows,
    is_reader_media_artifact,
    reader_media_plot_type_labels,
    semantic_kind_label,
    time_selected_label,
)
from .reader_evidence_visual import render_notebook_reader_evidence_artifact_visual


def discover_reader_evidence_manifests(workdir: str | Path) -> list[dict[str, Any]]:
    """Return small inventory rows for round-local Reader evidence manifests."""

    root = Path(workdir)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("inputs/r*/reader_evidence*.json")):
        rows.append(_reader_evidence_manifest_row(path, workdir=root))
    return rows


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
                plot_type_label = semantic_kind_label(semantic_kind)
                artifact_label = " | ".join(
                    part
                    for part in (
                        round_label,
                        reader_experiment_id,
                        design_id,
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
                    "selected_binding": dict(mapping(item.get("selected_binding"))),
                    "binding_source": dict(mapping(item.get("binding_source"))),
                    "reader_config_path": str(item.get("reader_config_path") or ""),
                    "reader_record_id": str(item.get("reader_record_id") or ""),
                    "sequence": item.get("sequence") or "",
                    "synthesis_name": item.get("synthesis_name") or "",
                    "semantic_kind": semantic_kind,
                    "plot_type_label": plot_type_label,
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


def build_notebook_reader_evidence_rows(view_model: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return notebook-facing rows for Reader evidence manifest status."""

    campaign = mapping(view_model.get("campaign"))
    workdir = campaign.get("workdir")
    rows = []
    for row in sequence(view_model.get("reader_evidence")):
        item = mapping(row)
        rows.append(
            {
                "status": item.get("status") or "unknown",
                "round": item.get("round") or "",
                "rows": item.get("rows") or 0,
                "distinct_ids": item.get("distinct_ids") or 0,
                "reader_experiments": item.get("reader_experiments") or 0,
                "artifact_count": item.get("artifact_count") or 0,
                "missing_artifact_rows": item.get("missing_artifact_rows") or 0,
                "path": compact_path(item.get("path"), base=workdir),
            }
        )
    return rows


def build_notebook_reader_evidence_artifact_rows(view_model: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return notebook-facing flattened Reader artifact rows."""

    rows = []
    for row in sequence(view_model.get("reader_evidence_artifacts")):
        item = mapping(row)
        output = {
            "label": item.get("label") or "",
            "round": item.get("round") or "",
            "id": item.get("id") or "",
            "candidate_id": item.get("candidate_id") or item.get("id") or "",
            "design_id": item.get("design_id") or "",
            "reader_experiment_id": item.get("reader_experiment_id") or "",
            "reduction_id": item.get("reduction_id") or "",
            "evidence_role": item.get("evidence_role") or "",
            "claim_status": item.get("claim_status") or "",
            "selected_binding": dict(mapping(item.get("selected_binding"))),
            "binding_source": dict(mapping(item.get("binding_source"))),
            "reader_config_path": item.get("reader_config_path") or "",
            "reader_record_id": item.get("reader_record_id") or "",
            "sequence": item.get("sequence") or "",
            "synthesis_name": item.get("synthesis_name") or "",
            "semantic_kind": item.get("semantic_kind") or "",
            "plot_type_label": item.get("plot_type_label") or semantic_kind_label(item.get("semantic_kind")),
            "kind": item.get("kind") or "",
            "artifact_record_id": item.get("artifact_record_id") or "",
            "scope": item.get("scope") or "",
            "exists": bool(item.get("exists")),
            "media_type": item.get("media_type") or "",
            "bytes": item.get("bytes"),
            "sha256": item.get("sha256") or "",
            "source_manifest_sha256": item.get("source_manifest_sha256") or "",
            "manifest_path": item.get("manifest_path") or "",
            "path": item.get("path") or "",
            "path_label": item.get("path_label") or item.get("path") or "",
        }
        if "time_selected_h" in item:
            output["time_selected_h"] = _blank_if_none(item.get("time_selected_h"))
        if "source_manifest_path" in item:
            output["source_manifest_path"] = item.get("source_manifest_path") or ""
        rows.append(output)
    return rows


def _blank_if_none(value: object) -> object:
    return "" if value is None else value


def build_notebook_reader_evidence_surface(view_model: Mapping[str, Any]) -> dict[str, Any]:
    """Return the Reader evidence rows and media choices for notebook rendering."""

    rows = build_notebook_reader_evidence_rows(view_model)
    artifact_rows = build_notebook_reader_evidence_artifact_rows(view_model)
    media_rows = dedupe_reader_media_labels([row for row in artifact_rows if is_reader_media_artifact(row)])
    return {
        "rows": rows,
        "artifact_rows": artifact_rows,
        "media_rows": media_rows,
        "media_labels": [str(row["label"]) for row in media_rows],
        "media_plot_type_labels": reader_media_plot_type_labels(media_rows),
    }


def build_notebook_reader_evidence_plot_type_options(surface: Mapping[str, Any]) -> list[str]:
    """Return notebook dropdown labels for Reader media plot types."""

    return [str(label) for label in sequence(surface.get("media_plot_type_labels")) if str(label).strip()]


def build_notebook_reader_evidence_visual_choices(surface: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return Reader media plot types as first-class notebook deliverable choices."""

    return [
        {
            "label": f"Reader evidence | {label}",
            "title": label,
            "surface_kind": "reader_evidence",
            "reader_plot_type_label": label,
        }
        for label in build_notebook_reader_evidence_plot_type_options(surface)
    ]


def build_notebook_reader_evidence_artifact_options(
    surface: Mapping[str, Any],
    *,
    selected_plot_type_label: str | None,
) -> list[str]:
    """Return artifact dropdown labels scoped to the selected Reader plot type."""

    media_rows = filter_reader_media_rows(surface, selected_plot_type_label=selected_plot_type_label)
    return [str(row.get("label") or "") for row in media_rows if str(row.get("label") or "").strip()]


def render_notebook_reader_evidence_plot_type_control(surface: Mapping[str, Any], *, mo: Any) -> Any | None:
    """Render the plot-type dropdown for generated notebooks."""

    options = build_notebook_reader_evidence_plot_type_options(surface)
    if not options:
        return None
    return mo.ui.dropdown(options, value=options[0], label="Reader plot type", full_width=True)


def render_notebook_reader_evidence_artifact_control(
    surface: Mapping[str, Any],
    *,
    selected_plot_type_label: str | None,
    mo: Any,
) -> Any | None:
    """Render the plot-instance dropdown scoped to a plot type."""

    options = build_notebook_reader_evidence_artifact_options(
        surface,
        selected_plot_type_label=selected_plot_type_label,
    )
    if not options:
        return None
    return mo.ui.dropdown(options, value=options[0], label="Reader plot instance", searchable=True, full_width=True)


def render_notebook_reader_evidence_panel(
    view_model: Mapping[str, Any], *, mo: Any, opal_table: Any, pl: Any
) -> dict[str, Any]:
    """Render the Reader evidence summary panel for generated marimo notebooks."""

    surface = build_notebook_reader_evidence_surface(view_model)
    rows = sequence(surface.get("rows"))
    artifact_rows = sequence(surface.get("artifact_rows"))

    evidence_panel = opal_table(pl.DataFrame(rows), page_size=8) if rows else mo.md("No Reader evidence.")
    artifact_table = (
        opal_table(
            pl.DataFrame(
                [{key: value for key, value in mapping(row).items() if key != "path"} for row in artifact_rows]
            ),
            page_size=8,
        )
        if artifact_rows
        else mo.md("No Reader artifact rows.")
    )
    return {
        "panel": mo.vstack([evidence_panel, artifact_table], gap=0.35),
        "surface": surface,
    }


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


__all__ = [
    "build_notebook_reader_evidence_artifact_rows",
    "build_notebook_reader_evidence_artifact_options",
    "build_notebook_reader_evidence_plot_type_options",
    "build_notebook_reader_evidence_rows",
    "build_notebook_reader_evidence_surface",
    "build_notebook_reader_evidence_visual_choices",
    "discover_reader_evidence_artifacts",
    "discover_reader_evidence_manifests",
    "render_notebook_reader_evidence_artifact_visual",
    "render_notebook_reader_evidence_artifact_control",
    "render_notebook_reader_evidence_panel",
    "render_notebook_reader_evidence_plot_type_control",
]
