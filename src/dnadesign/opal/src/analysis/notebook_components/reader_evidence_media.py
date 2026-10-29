"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/reader_evidence_media.py

Media helpers for Reader evidence notebook components.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from dnadesign.opal.api.reader_evidence import optional_reader_evidence_artifact_adapter

from ._support import display_name, mapping, sequence


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


def preferred_reader_media_rows(media_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one preferred inline representation per logical Reader deliverable."""

    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    order: list[tuple[str, ...]] = []
    for row in media_rows:
        item = dict(row)
        key = _reader_media_instance_key(item)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(item)
    preferred: list[dict[str, Any]] = []
    for key in order:
        choices = grouped[key]
        variants = [_reader_media_variant(item) for item in choices]
        duplicated = sorted({variant for variant in variants if variants.count(variant) > 1})
        if duplicated:
            raise ValueError("Reader evidence record publishes duplicate media variants: " + ", ".join(duplicated))
        choices = sorted(choices, key=_inline_media_preference)
        selected = dict(choices[0])
        selected["available_media"] = [
            {
                "media_type": str(item.get("media_type") or ""),
                "path": str(item.get("path") or ""),
                "path_label": str(item.get("path_label") or item.get("path") or ""),
                "sha256": str(item.get("sha256") or ""),
            }
            for item in choices
        ]
        preferred.append(selected)
    return preferred


def _reader_media_instance_key(row: Mapping[str, Any]) -> tuple[str, ...]:
    def key_text(field: str) -> str:
        value = row.get(field)
        return "" if value is None else str(value)

    return (
        key_text("manifest_path"),
        key_text("round"),
        key_text("id"),
        key_text("candidate_id"),
        key_text("design_id"),
        key_text("reader_experiment_id"),
        key_text("reduction_id"),
        key_text("semantic_kind"),
        key_text("time_selected_h"),
    )


def _reader_media_variant(row: Mapping[str, Any]) -> str:
    media_type = str(row.get("media_type") or "").strip().lower()
    if media_type:
        return media_type
    suffix = Path(str(row.get("path") or "")).suffix.lower()
    return suffix or "unknown"


def _inline_media_preference(row: Mapping[str, Any]) -> tuple[int, str]:
    media_type = str(row.get("media_type") or "").lower()
    suffix = Path(str(row.get("path") or "")).suffix.lower()
    if media_type == "image/png" or suffix == ".png":
        priority = 0
    elif media_type.startswith("image/") or suffix in {".jpg", ".jpeg"}:
        priority = 1
    elif media_type == "application/pdf" or suffix == ".pdf":
        priority = 2
    else:
        priority = 3
    return priority, str(row.get("path") or "")


def semantic_kind_label(value: Any) -> str:
    """Return the notebook label for a Reader artifact semantic kind."""

    semantic_kind = str(value or "artifact").strip()
    adapter = optional_reader_evidence_artifact_adapter(semantic_kind)
    if adapter is not None and adapter.display_label is not None:
        return adapter.display_label
    return semantic_kind.replace("_", " ").strip().title() or "Reader artifact"


def reader_round_display_label(value: Any) -> str:
    """Return a compact human label while retaining the raw round in its row."""

    token = str(value or "").strip()
    match = re.fullmatch(r"r(\d+)", token, flags=re.IGNORECASE)
    return f"Round {match.group(1)}" if match else display_name(token)


def reader_experiment_display_label(value: Any) -> str:
    """Humanize a Reader experiment slug without assigning assay semantics."""

    token = str(value or "").strip()
    if not token:
        return ""
    match = re.fullmatch(r"(\d{8})[_-](.+)", token)
    if match is None:
        return display_name(token)
    try:
        day = datetime.strptime(match.group(1), "%Y%m%d").date().isoformat()
    except ValueError:
        return display_name(token)
    suffix = display_name(re.sub(r"(?<=\d)-(?=\d)", "–", match.group(2)))
    return f"{day} · {suffix}" if suffix else day


def reader_reduction_display_label(value: Any) -> str:
    """Humanize a Reader reduction ID without changing its contract value."""

    token = str(value or "").strip()
    if not token:
        return ""
    if token.startswith("event_"):
        match = re.search(r"_(\d+(?:p\d+)?)_(\d+(?:p\d+)?)h_post$", token)
        if match is not None:
            start, end = (part.replace("p", ".") for part in match.groups())
            return f"{start}–{end} h post-event"
    return display_name(token)


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
    "preferred_reader_media_rows",
    "reader_media_plot_type_labels",
    "reader_experiment_display_label",
    "reader_reduction_display_label",
    "reader_round_display_label",
    "select_reader_media_artifact",
    "semantic_kind_label",
    "time_selected_label",
]
