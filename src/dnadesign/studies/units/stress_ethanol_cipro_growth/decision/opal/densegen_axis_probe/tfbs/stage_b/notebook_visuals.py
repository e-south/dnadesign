"""Register TFBS Stage B realized-label plots as OPAL notebook visuals."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from ..stage_a.manifests import file_sha256
from .notebook_visual_specs import (
    realized_caption,
    realized_group_key,
    realized_metric_expression,
    realized_metric_label,
    realized_metric_name,
    realized_summary_name,
    realized_tidy_csv_path,
    realized_visual_id,
    realized_visual_label,
    slot_caption,
    slot_group_key,
    slot_metric_expression,
    slot_metric_label,
    slot_metric_name,
    slot_summary_name,
    slot_tidy_csv_path,
    slot_visual_id,
    slot_visual_label,
    slug_token,
)
from .review_plots import REALIZED_REVIEW_PLOT_MANIFEST_SCHEMA_VERSION
from .slot_plots import SLOT_DIAGNOSTIC_PLOT_MANIFEST_SCHEMA_VERSION

COLLECTION_VISUAL_MANIFEST_INDEX_SCHEMA_VERSION = "opal.collection_visual_manifest_index.v1"
REALIZED_REVIEW_SURFACE_KIND = "study_realized_label_review"
REALIZED_REVIEW_COMPARISON_SET_PREFIX = "stage_b_realized_label_review"
SLOT_DIAGNOSTIC_SURFACE_KIND = "study_slot_count_confound_diagnostic"
SLOT_DIAGNOSTIC_COMPARISON_SET_KEY = "stage_b_slot_count_diagnostics"
SLOT_DIAGNOSTIC_COMPARISON_SET_LABEL = "Stage B slot count diagnostics"


def register_tfbs_stage_b_realized_review_visuals(
    *,
    collection_visual_index_path: str | Path,
    plot_manifest_json_path: str | Path,
    trajectory_csv_path: str | Path,
    pair_summary_csv_path: str | Path,
) -> dict[str, Any]:
    """Add realized-label review plots to an existing OPAL collection visual index."""

    index_path = Path(collection_visual_index_path)
    plot_manifest_path = Path(plot_manifest_json_path)
    trajectory_path = Path(trajectory_csv_path)
    pair_path = Path(pair_summary_csv_path)
    index = _read_collection_visual_index(index_path)
    plot_manifest = _read_realized_plot_manifest(plot_manifest_path)
    _require_existing_file(trajectory_path, role="trajectory CSV")
    _require_existing_file(pair_path, role="pair summary CSV")

    retained_visuals = [
        visual
        for visual in _mapping_list(index.get("visuals"), field="visuals")
        if not _is_owned_realized_review_visual(visual)
    ]
    retained_sets = [
        item
        for item in _mapping_list(index.get("comparison_sets"), field="comparison_sets")
        if not _is_owned_realized_review_set(item)
    ]
    realized_visuals = _visual_entries(
        plot_manifest=plot_manifest,
        plot_manifest_path=plot_manifest_path,
        trajectory_csv_path=trajectory_path,
        pair_summary_csv_path=pair_path,
    )
    refreshed = {
        **index,
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "comparison_sets": [*_comparison_set_entries(realized_visuals), *retained_sets],
        "visuals": [*realized_visuals, *retained_visuals],
    }
    refreshed["comparison_set_count"] = len(refreshed["comparison_sets"])
    refreshed["visual_count"] = len(refreshed["visuals"])
    index_path.write_text(json.dumps(refreshed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "status": "REGISTERED",
        "collection_visual_index_path": str(index_path),
        "collection_visual_index_hash": file_sha256(index_path),
        "registered_visual_count": len(realized_visuals),
        "comparison_set_keys": [item["key"] for item in _comparison_set_entries(realized_visuals)],
    }


def maybe_register_tfbs_stage_b_realized_review_visuals(
    *,
    config_manifest_path: str | Path,
    plot_manifest_json_path: str | Path,
    trajectory_csv_path: str | Path,
    pair_summary_csv_path: str | Path,
    collection_visual_index_path: str | Path | None = None,
) -> dict[str, Any]:
    """Register review plots when a notebook collection visual index exists."""

    explicit_index = Path(collection_visual_index_path) if collection_visual_index_path is not None else None
    default_index = _default_collection_visual_index_path(Path(config_manifest_path))
    index_path = explicit_index or default_index
    if not index_path.exists():
        if explicit_index is not None:
            raise FileNotFoundError(f"Stage B notebook collection visual index not found: {index_path}")
        return {
            "status": "SKIPPED_INDEX_NOT_FOUND",
            "collection_visual_index_path": str(index_path),
            "registered_visual_count": 0,
        }
    return register_tfbs_stage_b_realized_review_visuals(
        collection_visual_index_path=index_path,
        plot_manifest_json_path=plot_manifest_json_path,
        trajectory_csv_path=trajectory_csv_path,
        pair_summary_csv_path=pair_summary_csv_path,
    )


def register_tfbs_stage_b_slot_diagnostic_visuals(
    *,
    collection_visual_index_path: str | Path,
    plot_manifest_json_path: str | Path,
    trajectory_csv_path: str | Path,
    pair_summary_csv_path: str | Path,
    count_distribution_csv_path: str | Path,
) -> dict[str, Any]:
    """Add slot-count diagnostic plots to an existing OPAL collection visual index."""

    index_path = Path(collection_visual_index_path)
    plot_manifest_path = Path(plot_manifest_json_path)
    trajectory_path = Path(trajectory_csv_path)
    pair_path = Path(pair_summary_csv_path)
    distribution_path = Path(count_distribution_csv_path)
    index = _read_collection_visual_index(index_path)
    plot_manifest = _read_slot_plot_manifest(plot_manifest_path)
    _require_existing_file(trajectory_path, role="slot trajectory CSV")
    _require_existing_file(pair_path, role="slot pair summary CSV")
    _require_existing_file(distribution_path, role="slot count distribution CSV")

    retained_visuals = [
        visual
        for visual in _mapping_list(index.get("visuals"), field="visuals")
        if not _is_owned_slot_diagnostic_visual(visual)
    ]
    retained_sets = [
        item
        for item in _mapping_list(index.get("comparison_sets"), field="comparison_sets")
        if str(item.get("key") or "") != SLOT_DIAGNOSTIC_COMPARISON_SET_KEY
    ]
    slot_visuals = _slot_visual_entries(
        plot_manifest=plot_manifest,
        plot_manifest_path=plot_manifest_path,
        trajectory_csv_path=trajectory_path,
        pair_summary_csv_path=pair_path,
        count_distribution_csv_path=distribution_path,
    )
    refreshed = {
        **index,
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "comparison_sets": [
            *retained_sets,
            {
                "key": SLOT_DIAGNOSTIC_COMPARISON_SET_KEY,
                "label": SLOT_DIAGNOSTIC_COMPARISON_SET_LABEL,
                "match": {"review_surface": "slot_count_diagnostics"},
            },
        ],
        "visuals": [*retained_visuals, *slot_visuals],
    }
    refreshed["comparison_set_count"] = len(refreshed["comparison_sets"])
    refreshed["visual_count"] = len(refreshed["visuals"])
    index_path.write_text(json.dumps(refreshed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "status": "REGISTERED",
        "collection_visual_index_path": str(index_path),
        "collection_visual_index_hash": file_sha256(index_path),
        "registered_visual_count": len(slot_visuals),
        "comparison_set_key": SLOT_DIAGNOSTIC_COMPARISON_SET_KEY,
    }


def maybe_register_tfbs_stage_b_slot_diagnostic_visuals(
    *,
    config_manifest_path: str | Path,
    plot_manifest_json_path: str | Path,
    trajectory_csv_path: str | Path,
    pair_summary_csv_path: str | Path,
    count_distribution_csv_path: str | Path,
    collection_visual_index_path: str | Path | None = None,
) -> dict[str, Any]:
    """Register slot diagnostics when a notebook collection visual index exists."""

    explicit_index = Path(collection_visual_index_path) if collection_visual_index_path is not None else None
    default_index = _default_collection_visual_index_path(Path(config_manifest_path))
    index_path = explicit_index or default_index
    if not index_path.exists():
        if explicit_index is not None:
            raise FileNotFoundError(f"Stage B notebook collection visual index not found: {index_path}")
        return {
            "status": "SKIPPED_INDEX_NOT_FOUND",
            "collection_visual_index_path": str(index_path),
            "registered_visual_count": 0,
        }
    return register_tfbs_stage_b_slot_diagnostic_visuals(
        collection_visual_index_path=index_path,
        plot_manifest_json_path=plot_manifest_json_path,
        trajectory_csv_path=trajectory_csv_path,
        pair_summary_csv_path=pair_summary_csv_path,
        count_distribution_csv_path=count_distribution_csv_path,
    )


def _default_collection_visual_index_path(config_manifest_path: Path) -> Path:
    if config_manifest_path.parent.name == "manifests":
        stage_b_root = config_manifest_path.parent.parent
    else:
        stage_b_root = config_manifest_path.parent
    return stage_b_root / "notebooks" / "collection_visuals" / "collection_visual_manifest.json"


def _read_collection_visual_index(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != COLLECTION_VISUAL_MANIFEST_INDEX_SCHEMA_VERSION:
        raise ValueError(f"Unsupported OPAL collection visual index schema: {payload.get('schema_version')!r}")
    _mapping_list(payload.get("visuals"), field="visuals")
    _mapping_list(payload.get("comparison_sets"), field="comparison_sets")
    return payload


def _read_realized_plot_manifest(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != REALIZED_REVIEW_PLOT_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Unsupported Stage B realized review plot manifest schema: {payload.get('schema_version')!r}")
    plots = _mapping_list(payload.get("plots"), field="plots")
    if not plots:
        raise ValueError("Stage B realized review plot manifest contains no plots")
    return payload


def _read_slot_plot_manifest(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("schema_version") != SLOT_DIAGNOSTIC_PLOT_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Unsupported Stage B slot diagnostic plot manifest schema: {payload.get('schema_version')!r}")
    plots = _mapping_list(payload.get("plots"), field="plots")
    if not plots:
        raise ValueError("Stage B slot diagnostic plot manifest contains no plots")
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _mapping_list(value: Any, *, field: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"OPAL collection visual index field {field!r} must be a list")
    if not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"OPAL collection visual index field {field!r} must contain objects")
    return list(value)


def _is_owned_realized_review_visual(visual: Mapping[str, Any]) -> bool:
    return str(visual.get("surface_kind") or "") == REALIZED_REVIEW_SURFACE_KIND


def _is_owned_realized_review_set(item: Mapping[str, Any]) -> bool:
    return str(item.get("key") or "").startswith(f"{REALIZED_REVIEW_COMPARISON_SET_PREFIX}__")


def _is_owned_slot_diagnostic_visual(visual: Mapping[str, Any]) -> bool:
    return (
        str(visual.get("surface_kind") or "") == SLOT_DIAGNOSTIC_SURFACE_KIND
        and str(visual.get("comparison_set_key") or "") == SLOT_DIAGNOSTIC_COMPARISON_SET_KEY
    )


def _visual_entries(
    *,
    plot_manifest: Mapping[str, Any],
    plot_manifest_path: Path,
    trajectory_csv_path: Path,
    pair_summary_csv_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for plot in _mapping_list(plot_manifest.get("plots"), field="plots"):
        plot_path = Path(str(plot.get("path") or ""))
        _require_existing_file(plot_path, role="review plot")
        kind = str(plot.get("kind") or "")
        label_name = str(plot.get("label_name") or "").strip()
        if not label_name:
            raise ValueError("Stage B realized review plot is missing label_name")
        tidy_path = realized_tidy_csv_path(
            kind=kind,
            trajectory_csv_path=trajectory_csv_path,
            pair_summary_csv_path=pair_summary_csv_path,
        )
        row_count = _row_count(
            kind=kind,
            trajectory_csv_path=trajectory_csv_path,
            pair_summary_csv_path=pair_summary_csv_path,
        )
        rows.append(
            {
                "visual_id": realized_visual_id(kind, label_name=label_name),
                "label": realized_visual_label(kind),
                "title": str(plot.get("title") or realized_visual_label(kind)),
                "target": label_name,
                "surface_kind": REALIZED_REVIEW_SURFACE_KIND,
                "kind": REALIZED_REVIEW_SURFACE_KIND,
                "view_kind": kind,
                "source_plot_name": kind,
                "source_plot_kind": REALIZED_REVIEW_SURFACE_KIND,
                "comparison_scope": "study_review",
                "comparison_set_key": _realized_comparison_set_key(label_name),
                "comparison_set_label": _realized_comparison_set_label(label_name),
                "comparison_set_match": {"review_surface": "realized_label_review", "label_name": label_name},
                "relationship_id": "positive_vs_null",
                "relationship_kind": "control_pair",
                "group_key": realized_group_key(kind),
                "metric": realized_metric_name(kind),
                "metric_label": realized_metric_label(kind),
                "metric_expression": realized_metric_expression(kind),
                "cohort": "selected",
                "summary": realized_summary_name(kind),
                "interval_kind": "none",
                "interpretation_note": plot_manifest.get("interpretation_boundary"),
                "row_count": row_count,
                "path": str(plot_path),
                "manifest_path": str(plot_manifest_path),
                "tidy_csv": str(tidy_path),
                "freshness": {"status": "current"},
                "caption": realized_caption(kind),
                "alt_text": str(plot.get("alt_text") or realized_caption(kind)),
            }
        )
    return rows


def _comparison_set_entries(visuals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for visual in visuals:
        key = str(visual["comparison_set_key"])
        entries[key] = {
            "key": key,
            "label": str(visual["comparison_set_label"]),
            "match": dict(visual["comparison_set_match"]),
        }
    return [entries[key] for key in sorted(entries)]


def _realized_comparison_set_key(label_name: str) -> str:
    return f"{REALIZED_REVIEW_COMPARISON_SET_PREFIX}__{slug_token(label_name)}"


def _realized_comparison_set_label(label_name: str) -> str:
    return f"{label_name} positive/null pair"


def _slot_visual_entries(
    *,
    plot_manifest: Mapping[str, Any],
    plot_manifest_path: Path,
    trajectory_csv_path: Path,
    pair_summary_csv_path: Path,
    count_distribution_csv_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for plot in _mapping_list(plot_manifest.get("plots"), field="plots"):
        plot_path = Path(str(plot.get("path") or ""))
        _require_existing_file(plot_path, role="slot diagnostic plot")
        kind = str(plot.get("kind") or "")
        tidy_path = slot_tidy_csv_path(
            kind=kind,
            trajectory_csv_path=trajectory_csv_path,
            pair_summary_csv_path=pair_summary_csv_path,
            count_distribution_csv_path=count_distribution_csv_path,
        )
        rows.append(
            {
                "visual_id": slot_visual_id(kind),
                "label": slot_visual_label(kind),
                "title": str(plot.get("title") or slot_visual_label(kind)),
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
                "group_key": slot_group_key(kind),
                "metric": slot_metric_name(kind),
                "metric_label": slot_metric_label(kind),
                "metric_expression": slot_metric_expression(kind),
                "cohort": "selected",
                "summary": slot_summary_name(kind),
                "interval_kind": "none",
                "interpretation_note": plot_manifest.get("interpretation_boundary"),
                "row_count": _csv_row_count(tidy_path),
                "path": str(plot_path),
                "manifest_path": str(plot_manifest_path),
                "tidy_csv": str(tidy_path),
                "freshness": {"status": "current"},
                "caption": slot_caption(kind),
                "alt_text": str(plot.get("alt_text") or slot_caption(kind)),
            }
        )
    return rows


def _require_existing_file(path: Path, *, role: str) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Stage B realized review {role} not found: {path}")


def _row_count(*, kind: str, trajectory_csv_path: Path, pair_summary_csv_path: Path) -> int:
    path = realized_tidy_csv_path(
        kind=kind,
        trajectory_csv_path=trajectory_csv_path,
        pair_summary_csv_path=pair_summary_csv_path,
    )
    with path.open("r", encoding="utf-8") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def _csv_row_count(path: Path) -> int:
    _require_existing_file(path, role="diagnostic tidy CSV")
    with path.open("r", encoding="utf-8") as handle:
        return max(0, sum(1 for _ in handle) - 1)
