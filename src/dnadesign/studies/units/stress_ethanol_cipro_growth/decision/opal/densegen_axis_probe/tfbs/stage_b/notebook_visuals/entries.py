"""Build OPAL collection visual entries from Stage B plot manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    REALIZED_REVIEW_COMPARISON_SET_PREFIX,
    REALIZED_REVIEW_SURFACE_KIND,
    SLOT_DIAGNOSTIC_COMPARISON_SET_KEY,
    SLOT_DIAGNOSTIC_COMPARISON_SET_LABEL,
    SLOT_DIAGNOSTIC_SURFACE_KIND,
)
from .io import csv_row_count, mapping_list, require_existing_file
from .specs import realized_visual_spec, slot_visual_spec, slug_token


def visual_entries(
    *,
    plot_manifest: Mapping[str, Any],
    plot_manifest_path: Path,
    trajectory_csv_path: Path,
    pair_summary_csv_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for plot in mapping_list(plot_manifest.get("plots"), field="plots"):
        plot_path = Path(str(plot.get("path") or ""))
        require_existing_file(plot_path, role="review plot")
        kind = str(plot.get("kind") or "")
        label_name = str(plot.get("label_name") or "").strip()
        if not label_name:
            raise ValueError("Stage B realized review plot is missing label_name")
        spec = realized_visual_spec(kind)
        tidy_path = spec.tidy_csv_path(
            trajectory_csv_path=trajectory_csv_path,
            pair_summary_csv_path=pair_summary_csv_path,
        )
        rows.append(
            {
                "visual_id": spec.visual_id(label_name=label_name),
                "label": spec.label,
                "title": str(plot.get("title") or spec.label),
                "target": label_name,
                "surface_kind": REALIZED_REVIEW_SURFACE_KIND,
                "kind": REALIZED_REVIEW_SURFACE_KIND,
                "view_kind": kind,
                "source_plot_name": kind,
                "source_plot_kind": REALIZED_REVIEW_SURFACE_KIND,
                "comparison_scope": "study_review",
                "comparison_set_key": realized_comparison_set_key(label_name),
                "comparison_set_label": realized_comparison_set_label(label_name),
                "comparison_set_match": {"review_surface": "realized_label_review", "label_name": label_name},
                "relationship_id": "positive_vs_null",
                "relationship_kind": "control_pair",
                "group_key": spec.group_key,
                "metric": spec.metric_name,
                "metric_label": spec.metric_label,
                "metric_expression": spec.metric_expression,
                "cohort": "selected",
                "summary": spec.summary_name,
                "interval_kind": "none",
                "interpretation_note": plot_manifest.get("interpretation_boundary"),
                "row_count": csv_row_count(tidy_path, role="realized review tidy CSV"),
                "path": str(plot_path),
                "manifest_path": str(plot_manifest_path),
                "tidy_csv": str(tidy_path),
                "freshness": {"status": "current"},
                "caption": spec.caption,
                "alt_text": str(plot.get("alt_text") or spec.caption),
            }
        )
    return rows


def comparison_set_entries(visuals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for visual in visuals:
        key = str(visual["comparison_set_key"])
        entries[key] = {
            "key": key,
            "label": str(visual["comparison_set_label"]),
            "match": dict(visual["comparison_set_match"]),
        }
    return [entries[key] for key in sorted(entries)]


def realized_comparison_set_key(label_name: str) -> str:
    return f"{REALIZED_REVIEW_COMPARISON_SET_PREFIX}__{slug_token(label_name)}"


def realized_comparison_set_label(label_name: str) -> str:
    return f"{label_name} positive/null pair"


def slot_visual_entries(
    *,
    plot_manifest: Mapping[str, Any],
    plot_manifest_path: Path,
    trajectory_csv_path: Path,
    pair_summary_csv_path: Path,
    count_distribution_csv_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for plot in mapping_list(plot_manifest.get("plots"), field="plots"):
        plot_path = Path(str(plot.get("path") or ""))
        require_existing_file(plot_path, role="slot diagnostic plot")
        kind = str(plot.get("kind") or "")
        spec = slot_visual_spec(kind)
        tidy_path = spec.tidy_csv_path(
            trajectory_csv_path=trajectory_csv_path,
            pair_summary_csv_path=pair_summary_csv_path,
            count_distribution_csv_path=count_distribution_csv_path,
        )
        rows.append(
            {
                "visual_id": spec.visual_id(),
                "label": spec.label,
                "title": str(plot.get("title") or spec.label),
                "surface_kind": SLOT_DIAGNOSTIC_SURFACE_KIND,
                "kind": SLOT_DIAGNOSTIC_SURFACE_KIND,
                "view_kind": kind,
                "source_plot_name": kind,
                "source_plot_kind": SLOT_DIAGNOSTIC_SURFACE_KIND,
                "comparison_scope": "study_review",
                "comparison_set_key": SLOT_DIAGNOSTIC_COMPARISON_SET_KEY,
                "comparison_set_label": SLOT_DIAGNOSTIC_COMPARISON_SET_LABEL,
                "comparison_set_match": {"review_surface": "slot_count_diagnostics"},
                "relationship_id": "slot_count_confound",
                "relationship_kind": "diagnostic_control",
                "group_key": spec.group_key,
                "metric": spec.metric_name,
                "metric_label": spec.metric_label,
                "metric_expression": spec.metric_expression,
                "cohort": "selected",
                "summary": spec.summary_name,
                "interval_kind": "none",
                "interpretation_note": plot_manifest.get("interpretation_boundary"),
                "row_count": csv_row_count(tidy_path, role="slot diagnostic tidy CSV"),
                "path": str(plot_path),
                "manifest_path": str(plot_manifest_path),
                "tidy_csv": str(tidy_path),
                "freshness": {"status": "current"},
                "caption": spec.caption,
                "alt_text": str(plot.get("alt_text") or spec.caption),
            }
        )
    return rows
