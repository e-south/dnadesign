"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/reporting/notebook_set.py

Manifest-backed view-model helpers for generated OPAL campaign-set notebooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ..analysis.notebook_components import build_notebook_campaign_set_selection_overlap_choice
from ..core.utils import ExitCodes, OpalError, now_iso
from .campaign_collection import load_campaign_collection_manifest
from .collection_visual_index import load_collection_visual_manifest_index
from .notebook import build_notebook_view_model

NOTEBOOK_CAMPAIGN_SET_VIEW_MODEL_SCHEMA_VERSION = "opal.notebook_campaign_set_view_model.v1"


def build_campaign_set_notebook_view_model(
    config_paths: Iterable[str | Path],
    *,
    round_selector: str | None = "latest",
    run_id: str | None = None,
    collection_manifest_path: str | Path | None = None,
    collection_visual_index_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a manifest-backed view model for one or more OPAL campaigns."""

    paths = _validated_campaign_set_paths(config_paths)
    if len(paths) == 1 and (collection_manifest_path is not None or collection_visual_index_path is not None):
        raise OpalError(
            "Campaign collection inputs require at least two distinct campaign configs.",
            ExitCodes.BAD_ARGS,
        )
    if run_id is not None and len(paths) != 1:
        raise OpalError(
            "Campaign-set notebooks only support run_id pinning for a single campaign.",
            ExitCodes.BAD_ARGS,
        )
    campaigns = [
        build_notebook_view_model(
            path,
            round_selector=round_selector,
            run_id=run_id if len(paths) == 1 else None,
        )
        for path in paths
    ]
    warnings = [
        {**warning, "campaign_slug": campaign["campaign"]["slug"]}
        for campaign in campaigns
        for warning in (campaign.get("warnings") or [])
        if isinstance(warning, dict)
    ]
    collection = (
        load_campaign_collection_manifest(collection_manifest_path, campaigns)
        if collection_manifest_path is not None
        else None
    )
    collection_visual_index = (
        load_collection_visual_manifest_index(
            collection_visual_index_path,
            expected_collection_id=str(collection["collection_id"]) if collection is not None else None,
            allowed_surface_kinds=collection.get("collection_visual_surface_kinds") if collection is not None else None,
        )
        if collection_visual_index_path is not None
        else None
    )
    collection_visuals = (
        _notebook_collection_visuals(collection_visual_index, index_path=Path(collection_visual_index_path))
        if collection_visual_index is not None and collection_visual_index_path is not None
        else []
    )
    selection_overlap = (
        build_notebook_campaign_set_selection_overlap_choice(
            campaigns,
            round_selector=round_selector or "latest",
        )
        if _supports_campaign_selection_overlap(campaigns)
        else None
    )
    if selection_overlap is not None:
        collection_visuals.append(selection_overlap)
    return {
        "schema_version": NOTEBOOK_CAMPAIGN_SET_VIEW_MODEL_SCHEMA_VERSION,
        "generated_at": now_iso(),
        "round_selector": round_selector or "latest",
        "campaign_count": len(campaigns),
        "campaigns": campaigns,
        "collection": collection,
        "collection_visual_index": collection_visual_index,
        "collection_visuals": collection_visuals,
        "warnings": warnings,
    }


def build_campaign_set_round_options(config_paths: Iterable[str | Path]) -> list[str]:
    """Return stable round selector options available across a campaign set."""

    paths = _validated_campaign_set_paths(config_paths)
    round_indexes: set[int] = set()
    for path in paths:
        campaign = build_notebook_view_model(path, round_selector="all")
        progress = campaign.get("progress") or {}
        for row in progress.get("rounds") or []:
            if not isinstance(row, dict):
                raise OpalError(
                    "Campaign-set round options expected progress round rows to be objects.",
                    ExitCodes.CONTRACT_VIOLATION,
                )
            round_index = row.get("round_index")
            if round_index is None:
                raise OpalError(
                    "Campaign-set round options found a progress row without round_index.",
                    ExitCodes.CONTRACT_VIOLATION,
                )
            try:
                round_indexes.add(int(round_index))
            except (TypeError, ValueError) as exc:
                raise OpalError(
                    f"Campaign-set round options found a non-integer round_index: {round_index!r}.",
                    ExitCodes.CONTRACT_VIOLATION,
                ) from exc

    return ["latest", "all", *(str(index) for index in sorted(round_indexes))]


def _validated_campaign_set_paths(config_paths: Iterable[str | Path]) -> list[Path]:
    paths = [Path(path) for path in config_paths]
    if not paths:
        raise OpalError("Campaign notebooks require at least one campaign config.", ExitCodes.BAD_ARGS)

    resolved = [str(path.resolve()) for path in paths]
    duplicates = sorted({path for path in resolved if resolved.count(path) > 1})
    if duplicates:
        raise OpalError(
            "Campaign-set notebooks require distinct campaign configs; duplicates: " + ", ".join(duplicates),
            ExitCodes.BAD_ARGS,
        )
    return paths


def _supports_campaign_selection_overlap(campaigns: list[dict[str, Any]]) -> bool:
    """Return whether pooled overlap compares distinct campaigns, not target views."""

    if len(campaigns) < 2:
        return False
    return all(len((campaign.get("campaign") or {}).get("selection_views") or []) <= 1 for campaign in campaigns)


def _notebook_collection_visuals(index: dict[str, Any], *, index_path: Path) -> list[dict[str, Any]]:
    base_path = index_path.expanduser().resolve().parent
    visuals: list[dict[str, Any]] = []
    for raw in index.get("visuals") or []:
        visual = dict(raw)
        for key in ("path", "tidy_csv", "manifest_path"):
            if visual.get(key) not in (None, ""):
                visual[key] = str(_resolve_index_relative_path(visual[key], base_path=base_path))
        visuals.append(visual)
    return visuals


def _resolve_index_relative_path(value: object, *, base_path: Path) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = base_path / path
    return path.resolve(strict=False)
