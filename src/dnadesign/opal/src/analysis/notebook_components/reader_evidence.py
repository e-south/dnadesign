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
from typing import Any, Mapping

from ._support import compact_path, mapping, sequence
from .reader_evidence_discovery import (
    discover_reader_evidence_artifacts,
    discover_reader_evidence_manifests,
)
from .reader_evidence_media import (
    dedupe_reader_media_labels,
    filter_reader_media_rows,
    is_reader_media_artifact,
    preferred_reader_media_rows,
    reader_media_plot_type_labels,
    semantic_kind_label,
)
from .reader_evidence_visual import render_notebook_reader_evidence_artifact_visual


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
            "non_claim_boundary": item.get("non_claim_boundary") or "",
            "selected_binding": dict(mapping(item.get("selected_binding"))),
            "sources": dict(mapping(item.get("sources"))),
            "objective_overlay": (
                None if item.get("objective_overlay") is None else dict(mapping(item.get("objective_overlay")))
            ),
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
            "source_record_revision_digest": item.get("source_record_revision_digest") or "",
            "source_file_path": item.get("source_file_path") or "",
            "source_receipt_sha256": item.get("source_receipt_sha256") or "",
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
    media_rows = dedupe_reader_media_labels(
        preferred_reader_media_rows([row for row in artifact_rows if is_reader_media_artifact(row)])
    )
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
            "selection_scope": "campaign",
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


def build_notebook_reader_evidence_record_memory_key(
    *,
    campaign_slug: Any,
    reader_plot_type_label: Any,
) -> str:
    """Build a stable preference key for one campaign Reader deliverable."""

    scope = {
        "campaign_slug": str(campaign_slug or "").strip(),
        "reader_plot_type_label": str(reader_plot_type_label or "").strip(),
    }
    missing = [field for field, value in scope.items() if not value]
    if missing:
        raise ValueError(f"Reader evidence record memory scope is missing: {', '.join(missing)}.")
    return f"reader_evidence_record_v1:{json.dumps(scope, sort_keys=True, separators=(',', ':'))}"


def resolve_notebook_reader_evidence_preferred_record_label(
    record_labels: Any,
    *,
    preferred_record_label: Any | None,
) -> str:
    """Restore a remembered Reader record only while it remains available."""

    labels = [str(label) for label in sequence(record_labels) if str(label).strip()]
    if not labels:
        raise ValueError("Reader evidence record options must not be empty.")
    if len(labels) != len(set(labels)):
        raise ValueError("Reader evidence record labels must be unique.")
    preferred = str(preferred_record_label or "").strip()
    return preferred if preferred in labels else labels[0]


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
    preferred_record_label: Any | None = None,
    on_change: Any | None = None,
    mo: Any,
) -> Any | None:
    """Render the record dropdown scoped to one Reader deliverable."""

    options = build_notebook_reader_evidence_artifact_options(
        surface,
        selected_plot_type_label=selected_plot_type_label,
    )
    if not options:
        return None
    preferred = resolve_notebook_reader_evidence_preferred_record_label(
        options,
        preferred_record_label=preferred_record_label,
    )
    return mo.ui.dropdown(
        options,
        value=preferred,
        label="Reader record",
        searchable=True,
        full_width=True,
        on_change=on_change,
    )


def render_notebook_reader_evidence_record_control(
    surface: Mapping[str, Any],
    *,
    campaign_slug: Any,
    selected_plot_type_label: str | None,
    memory: Any,
    set_memory: Any,
    mo: Any,
) -> Any | None:
    """Render a record selector remembered by campaign and Reader deliverable."""

    plot_type_label = str(selected_plot_type_label or "").strip()
    if not plot_type_label:
        return None
    memory_key = build_notebook_reader_evidence_record_memory_key(
        campaign_slug=campaign_slug,
        reader_plot_type_label=plot_type_label,
    )

    def remember_record(value: Any) -> None:
        set_memory({**dict(memory()), memory_key: str(value)})

    return render_notebook_reader_evidence_artifact_control(
        surface,
        selected_plot_type_label=plot_type_label,
        preferred_record_label=dict(memory()).get(memory_key),
        on_change=remember_record,
        mo=mo,
    )


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


__all__ = [
    "build_notebook_reader_evidence_artifact_rows",
    "build_notebook_reader_evidence_artifact_options",
    "build_notebook_reader_evidence_plot_type_options",
    "build_notebook_reader_evidence_record_memory_key",
    "build_notebook_reader_evidence_rows",
    "build_notebook_reader_evidence_surface",
    "build_notebook_reader_evidence_visual_choices",
    "discover_reader_evidence_artifacts",
    "discover_reader_evidence_manifests",
    "render_notebook_reader_evidence_artifact_visual",
    "render_notebook_reader_evidence_artifact_control",
    "render_notebook_reader_evidence_panel",
    "render_notebook_reader_evidence_plot_type_control",
    "render_notebook_reader_evidence_record_control",
    "resolve_notebook_reader_evidence_preferred_record_label",
]
