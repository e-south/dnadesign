"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/reader_evidence_media.py

Media helpers for Reader evidence notebook components.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ._support import mapping, sequence

_SEMANTIC_KIND_LABELS = {
    "intensity_overview": "Time series + snapshot",
    "raw_kinetics": "Plate-reader time series",
    "sfxi_vec8_heatmap": "SFXI vec8 heatmap",
    "vec8_heatmap": "SFXI vec8 heatmap",
}


def is_reader_media_artifact(row: Mapping[str, Any]) -> bool:
    """Return true for Reader artifacts marimo can render inline."""

    if not row.get("exists"):
        return False
    media_type = str(row.get("media_type") or "")
    suffix = Path(str(row.get("path") or "")).suffix.lower()
    return (
        media_type == "application/pdf"
        or media_type.startswith("image/")
        or suffix in {".pdf", ".png", ".jpg", ".jpeg"}
    )


def select_reader_media_artifact(
    surface: Mapping[str, Any],
    *,
    selected_plot_type_label: str,
    selected_artifact_label: str,
) -> Mapping[str, Any] | None:
    """Select one Reader media artifact from a scoped notebook surface."""

    for row in filter_reader_media_rows(surface, selected_plot_type_label=selected_plot_type_label):
        item = mapping(row)
        if str(item.get("label") or "") == selected_artifact_label:
            return item
    return None


def filter_reader_media_rows(
    surface: Mapping[str, Any],
    *,
    selected_plot_type_label: str | None,
) -> list[Mapping[str, Any]]:
    """Filter Reader media artifacts by plot-type label."""

    requested = str(selected_plot_type_label or "").strip()
    rows = [mapping(row) for row in sequence(surface.get("media_rows"))]
    if not requested:
        return rows
    return [row for row in rows if str(row.get("plot_type_label") or "") == requested]


def reader_media_plot_type_labels(media_rows: list[dict[str, Any]]) -> list[str]:
    """Return unique Reader media plot-type labels in notebook order."""

    labels: list[str] = []
    seen: set[str] = set()
    for row in media_rows:
        label = str(row.get("plot_type_label") or semantic_kind_label(row.get("semantic_kind"))).strip()
        if label and label not in seen:
            labels.append(label)
            seen.add(label)
    return labels


def dedupe_reader_media_labels(media_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append path context when duplicate Reader artifact labels occur."""

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


def semantic_kind_label(value: Any) -> str:
    """Return the notebook label for a Reader artifact semantic kind."""

    semantic_kind = str(value or "artifact").strip()
    if semantic_kind in _SEMANTIC_KIND_LABELS:
        return _SEMANTIC_KIND_LABELS[semantic_kind]
    return semantic_kind.replace("_", " ").strip().title() or "Reader artifact"


def time_selected_label(value: Any) -> str:
    """Return a compact time-selected label for Reader evidence dropdowns."""

    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.2f} h"
    except (TypeError, ValueError):
        return str(value)


__all__ = [
    "dedupe_reader_media_labels",
    "filter_reader_media_rows",
    "is_reader_media_artifact",
    "reader_media_plot_type_labels",
    "select_reader_media_artifact",
    "semantic_kind_label",
    "time_selected_label",
]
