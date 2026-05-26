"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/reporting/notebook_set.py

Manifest-backed view-model helpers for generated OPAL campaign-set notebooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ..core.utils import ExitCodes, OpalError, now_iso
from .campaign_collection import load_campaign_collection_manifest
from .campaign_set_artifacts import load_collection_visual_manifest_index
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
        load_collection_visual_manifest_index(collection_visual_index_path)
        if collection_visual_index_path is not None
        else None
    )
    return {
        "schema_version": NOTEBOOK_CAMPAIGN_SET_VIEW_MODEL_SCHEMA_VERSION,
        "generated_at": now_iso(),
        "round_selector": round_selector or "latest",
        "campaign_count": len(campaigns),
        "campaigns": campaigns,
        "collection": collection,
        "collection_visual_index": collection_visual_index,
        "collection_visuals": list(collection_visual_index.get("visuals") or []) if collection_visual_index else [],
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
