"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/tfbs/stage_b/notebook_visuals/registration.py

Register TFBS Stage B study visuals into OPAL collection notebooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from ...stage_a.manifests import file_sha256
from .contracts import (
    REALIZED_REVIEW_COMPARISON_SET_PREFIX,
    REALIZED_REVIEW_SURFACE_KIND,
    SLOT_DIAGNOSTIC_COMPARISON_SET_KEY,
    SLOT_DIAGNOSTIC_COMPARISON_SET_LABEL,
    SLOT_DIAGNOSTIC_SURFACE_KIND,
)
from .entries import comparison_set_entries, slot_visual_entries, visual_entries
from .io import (
    mapping_list,
    read_collection_visual_index,
    read_realized_plot_manifest,
    read_slot_plot_manifest,
    require_existing_file,
)


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
    index = read_collection_visual_index(index_path)
    plot_manifest = read_realized_plot_manifest(plot_manifest_path)
    require_existing_file(trajectory_path, role="trajectory CSV")
    require_existing_file(pair_path, role="pair summary CSV")

    realized_visuals = visual_entries(
        plot_manifest=plot_manifest,
        plot_manifest_path=plot_manifest_path,
        trajectory_csv_path=trajectory_path,
        pair_summary_csv_path=pair_path,
    )
    retained_visuals = [
        visual
        for visual in mapping_list(index.get("visuals"), field="visuals")
        if not _is_owned_realized_review_visual(visual)
    ]
    retained_sets = [
        item
        for item in mapping_list(index.get("comparison_sets"), field="comparison_sets")
        if not _is_owned_realized_review_set(item)
    ]
    refreshed = {
        **index,
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "comparison_sets": [*comparison_set_entries(realized_visuals), *retained_sets],
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
        "comparison_set_keys": [item["key"] for item in comparison_set_entries(realized_visuals)],
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
    index_path = explicit_index or _default_collection_visual_index_path(Path(config_manifest_path))
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
    index = read_collection_visual_index(index_path)
    plot_manifest = read_slot_plot_manifest(plot_manifest_path)
    require_existing_file(trajectory_path, role="slot trajectory CSV")
    require_existing_file(pair_path, role="slot pair summary CSV")
    require_existing_file(distribution_path, role="slot count distribution CSV")

    slot_visuals = slot_visual_entries(
        plot_manifest=plot_manifest,
        plot_manifest_path=plot_manifest_path,
        trajectory_csv_path=trajectory_path,
        pair_summary_csv_path=pair_path,
        count_distribution_csv_path=distribution_path,
    )
    retained_visuals = [
        visual
        for visual in mapping_list(index.get("visuals"), field="visuals")
        if not _is_owned_slot_diagnostic_visual(visual)
    ]
    retained_sets = [
        item
        for item in mapping_list(index.get("comparison_sets"), field="comparison_sets")
        if str(item.get("key") or "") != SLOT_DIAGNOSTIC_COMPARISON_SET_KEY
    ]
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
    index_path = explicit_index or _default_collection_visual_index_path(Path(config_manifest_path))
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
    stage_b_root = (
        config_manifest_path.parent.parent
        if config_manifest_path.parent.name == "manifests"
        else config_manifest_path.parent
    )
    return stage_b_root / "notebooks" / "collection_visuals" / "collection_visual_manifest.json"


def _is_owned_realized_review_visual(visual: Mapping[str, Any]) -> bool:
    return str(visual.get("surface_kind") or "") == REALIZED_REVIEW_SURFACE_KIND


def _is_owned_realized_review_set(item: Mapping[str, Any]) -> bool:
    return str(item.get("key") or "").startswith(f"{REALIZED_REVIEW_COMPARISON_SET_PREFIX}__")


def _is_owned_slot_diagnostic_visual(visual: Mapping[str, Any]) -> bool:
    return (
        str(visual.get("surface_kind") or "") == SLOT_DIAGNOSTIC_SURFACE_KIND
        and str(visual.get("comparison_set_key") or "") == SLOT_DIAGNOSTIC_COMPARISON_SET_KEY
    )
