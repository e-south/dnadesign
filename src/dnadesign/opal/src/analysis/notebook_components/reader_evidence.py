"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/reader_evidence.py

Builds notebook rows for Reader evidence manifests staged with observed labels.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ._support import compact_path, mapping, sequence

READER_EVIDENCE_SCHEMA_VERSION = "stress_ethanol_cipro_growth.reader_evidence.v1"
READER_EVIDENCE_PDF_HEIGHT = "52vh"
READER_EVIDENCE_IMAGE_MAX_HEIGHT = "min(56vh, 640px)"

_SEMANTIC_KIND_LABELS = {
    "raw_kinetics": "Plate-reader time series",
    "sfxi_vec8_heatmap": "SFXI vec8 heatmap",
    "vec8_heatmap": "SFXI vec8 heatmap",
}


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
        if payload is None or payload.get("schema_version") != READER_EVIDENCE_SCHEMA_VERSION:
            continue
        for evidence_row in sequence(payload.get("rows")):
            item = mapping(evidence_row)
            for artifact in sequence(item.get("artifacts")):
                artifact_item = mapping(artifact)
                path = artifact_item.get("path")
                semantic_kind = str(artifact_item.get("semantic_kind") or "artifact")
                design_id = str(item.get("design_id") or "")
                reader_experiment_id = str(item.get("reader_experiment_id") or "")
                round_label = str(payload.get("round") or _round_label(manifest_path))
                plot_type_label = _semantic_kind_label(semantic_kind)
                artifact_label = " | ".join(
                    part
                    for part in (
                        round_label,
                        reader_experiment_id,
                        design_id,
                        _time_selected_label(item.get("time_selected_h")),
                    )
                    if part
                )
                rows.append(
                    {
                        "label": artifact_label,
                        "round": round_label,
                        "id": str(item.get("id") or ""),
                        "design_id": design_id,
                        "reader_experiment_id": reader_experiment_id,
                        "time_selected_h": item.get("time_selected_h"),
                        "semantic_kind": semantic_kind,
                        "plot_type_label": plot_type_label,
                        "path": str(path or ""),
                        "path_label": compact_path(path, max_parts=5),
                        "exists": bool(artifact_item.get("exists")),
                        "media_type": str(artifact_item.get("media_type") or ""),
                        "manifest_path": str(manifest_path),
                        "manifest_path_label": compact_path(manifest_path, base=root),
                    }
                )
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
        rows.append(
            {
                "label": item.get("label") or "",
                "round": item.get("round") or "",
                "id": item.get("id") or "",
                "design_id": item.get("design_id") or "",
                "reader_experiment_id": item.get("reader_experiment_id") or "",
                "time_selected_h": item.get("time_selected_h") or "",
                "semantic_kind": item.get("semantic_kind") or "",
                "plot_type_label": item.get("plot_type_label") or _semantic_kind_label(item.get("semantic_kind")),
                "exists": bool(item.get("exists")),
                "media_type": item.get("media_type") or "",
                "path": item.get("path") or "",
                "path_label": item.get("path_label") or item.get("path") or "",
            }
        )
    return rows


def build_notebook_reader_evidence_surface(view_model: Mapping[str, Any]) -> dict[str, Any]:
    """Return the Reader evidence rows and media choices for notebook rendering."""

    rows = build_notebook_reader_evidence_rows(view_model)
    artifact_rows = build_notebook_reader_evidence_artifact_rows(view_model)
    media_rows = _dedupe_reader_media_labels([row for row in artifact_rows if _is_reader_media_artifact(row)])
    return {
        "rows": rows,
        "artifact_rows": artifact_rows,
        "media_rows": media_rows,
        "media_labels": [str(row["label"]) for row in media_rows],
        "media_plot_type_labels": _reader_media_plot_type_labels(media_rows),
    }


def build_notebook_reader_evidence_plot_type_options(surface: Mapping[str, Any]) -> list[str]:
    """Return notebook dropdown labels for Reader media plot types."""

    return [str(label) for label in sequence(surface.get("media_plot_type_labels")) if str(label).strip()]


def build_notebook_reader_evidence_artifact_options(
    surface: Mapping[str, Any],
    *,
    selected_plot_type_label: str | None,
) -> list[str]:
    """Return artifact dropdown labels scoped to the selected Reader plot type."""

    media_rows = _filter_reader_media_rows(surface, selected_plot_type_label=selected_plot_type_label)
    return [str(row.get("label") or "") for row in media_rows if str(row.get("label") or "").strip()]


def render_notebook_reader_evidence_plot_type_control(surface: Mapping[str, Any], *, mo: Any) -> Any | None:
    """Render the Reader plot-type dropdown for generated notebooks."""

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
    """Render the Reader artifact dropdown scoped to a plot type."""

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


def render_notebook_reader_evidence_artifact_visual(
    surface: Mapping[str, Any],
    *,
    selected_plot_type_label: str | None,
    selected_artifact_label: str | None,
    mo: Any,
) -> Any:
    """Render the selected Reader PDF or image artifact for generated notebooks."""

    if not selected_plot_type_label:
        return mo.md("No Reader plot type selected.")
    if not selected_artifact_label:
        return mo.md("No Reader artifact selected.")
    selected = _select_reader_media_artifact(
        surface,
        selected_plot_type_label=selected_plot_type_label,
        selected_artifact_label=selected_artifact_label,
    )
    if selected is None:
        return mo.md("Selected Reader artifact is no longer available.")
    path = Path(str(selected.get("path") or ""))
    media_type = str(selected.get("media_type") or "")
    if not path.exists():
        return mo.md(f"Reader artifact missing: `{path}`")
    if media_type == "application/pdf" or path.suffix.lower() == ".pdf":
        return mo.pdf(path, width="100%", height=READER_EVIDENCE_PDF_HEIGHT)
    if media_type.startswith("image/") or path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        return mo.image(
            path.read_bytes(),
            alt=str(selected.get("label") or "Reader artifact"),
            caption=str(selected.get("label") or ""),
            rounded=True,
            style={
                "width": "auto",
                "max-height": READER_EVIDENCE_IMAGE_MAX_HEIGHT,
                "max-width": "100%",
                "height": "auto",
                "object-fit": "contain",
                "margin": "0 auto",
                "display": "block",
                "background": "white",
            },
        )
    return mo.md(f"Reader artifact: `{path}`")


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
    if payload.get("schema_version") != READER_EVIDENCE_SCHEMA_VERSION:
        return {**row, "status": "schema_attention"}
    summary = mapping(payload.get("summary"))
    row.update(
        {
            "round": str(payload.get("round") or round_label),
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


def _is_reader_media_artifact(row: Mapping[str, Any]) -> bool:
    if not row.get("exists"):
        return False
    media_type = str(row.get("media_type") or "")
    suffix = Path(str(row.get("path") or "")).suffix.lower()
    return (
        media_type == "application/pdf"
        or media_type.startswith("image/")
        or suffix in {".pdf", ".png", ".jpg", ".jpeg"}
    )


def _select_reader_media_artifact(
    surface: Mapping[str, Any],
    *,
    selected_plot_type_label: str,
    selected_artifact_label: str,
) -> Mapping[str, Any] | None:
    for row in _filter_reader_media_rows(surface, selected_plot_type_label=selected_plot_type_label):
        item = mapping(row)
        if str(item.get("label") or "") == selected_artifact_label:
            return item
    return None


def _filter_reader_media_rows(
    surface: Mapping[str, Any],
    *,
    selected_plot_type_label: str | None,
) -> list[Mapping[str, Any]]:
    requested = str(selected_plot_type_label or "").strip()
    rows = [mapping(row) for row in sequence(surface.get("media_rows"))]
    if not requested:
        return rows
    return [row for row in rows if str(row.get("plot_type_label") or "") == requested]


def _reader_media_plot_type_labels(media_rows: list[dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for row in media_rows:
        label = str(row.get("plot_type_label") or _semantic_kind_label(row.get("semantic_kind"))).strip()
        if label and label not in seen:
            labels.append(label)
            seen.add(label)
    return labels


def _dedupe_reader_media_labels(media_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = {}
    label_keys = [
        (str(row.get("plot_type_label") or "").strip(), str(row.get("label") or "").strip()) for row in media_rows
    ]
    duplicated = {key for key in label_keys if key[1] and label_keys.count(key) > 1}
    out: list[dict[str, Any]] = []
    for row in media_rows:
        item = dict(row)
        label = str(item.get("label") or "").strip() or str(item.get("path_label") or item.get("path") or "artifact")
        label_key = (str(item.get("plot_type_label") or "").strip(), label)
        if label_key in duplicated:
            counts[label_key] = counts.get(label_key, 0) + 1
            suffix = str(item.get("path_label") or item.get("path") or counts[label_key])
            label = f"{label} | {suffix}"
        item["label"] = label
        out.append(item)
    return out


def _semantic_kind_label(value: Any) -> str:
    semantic_kind = str(value or "artifact").strip()
    if semantic_kind in _SEMANTIC_KIND_LABELS:
        return _SEMANTIC_KIND_LABELS[semantic_kind]
    return semantic_kind.replace("_", " ").strip().title() or "Reader artifact"


def _time_selected_label(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.2f} h"
    except (TypeError, ValueError):
        return str(value)


def _read_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _round_label(path: Path) -> str:
    return path.parent.name if path.parent.name.startswith("r") else ""


__all__ = [
    "READER_EVIDENCE_SCHEMA_VERSION",
    "build_notebook_reader_evidence_artifact_rows",
    "build_notebook_reader_evidence_artifact_options",
    "build_notebook_reader_evidence_plot_type_options",
    "build_notebook_reader_evidence_rows",
    "build_notebook_reader_evidence_surface",
    "discover_reader_evidence_artifacts",
    "discover_reader_evidence_manifests",
    "render_notebook_reader_evidence_artifact_visual",
    "render_notebook_reader_evidence_artifact_control",
    "render_notebook_reader_evidence_panel",
    "render_notebook_reader_evidence_plot_type_control",
]
