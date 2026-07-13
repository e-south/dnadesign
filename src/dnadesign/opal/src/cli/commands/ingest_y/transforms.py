"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/cli/commands/ingest_y/transforms.py

Transform selection and plugin context setup for `opal ingest-y`.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ....core.round_context import PluginRegistryView, RoundCtx
from ....core.utils import OpalError, now_iso
from ....registries.transforms_y import get_transform_y
from .input_files import read_transform_params


def resolve_transform_settings(
    cfg: Any,
    *,
    transform_name_override: str | None,
    params_path: Path | None,
    input_path: Path,
) -> tuple[str, dict[str, Any]]:
    transform_name = (transform_name_override or cfg.data.transforms_y.name).strip()
    transform_params = cfg.data.transforms_y.params
    if params_path:
        transform_params = read_transform_params(params_path)

    # Default SFXI delta enforcement is the explicit Reader-to-OPAL hand-off.
    if transform_name == "sfxi_vec8_from_table_v1":
        transform_params = dict(transform_params or {})
        if "expected_log2_offset_delta" not in transform_params:
            deltas = {
                float(view.objective.params.get("intensity_log2_offset_delta", 0.0))
                for view in cfg.selection_views
                if view.objective.name == "sfxi_v1"
            }
            if len(deltas) > 1:
                raise OpalError("SFXI selection views must share intensity_log2_offset_delta at ingest.")
            transform_params["expected_log2_offset_delta"] = deltas.pop() if deltas else 0.0
        if "enforce_log2_offset_match" not in transform_params:
            transform_params["enforce_log2_offset_match"] = True
        if "sfxi_log_json" not in transform_params:
            candidate = input_path.parent / "sfxi_log.json"
            if candidate.exists():
                transform_params["sfxi_log_json"] = str(candidate)

    return transform_name, transform_params


def build_transform_context(cfg: Any, *, round_index: int, transform_name: str):
    registry = PluginRegistryView(
        model=cfg.model.name,
        objective="selection_views",
        selection="selection_views",
        transform_x=cfg.data.transforms_x.name,
        transform_y=transform_name,
    )
    round_ctx = RoundCtx(
        core={
            "core/run_id": f"ingest-{now_iso()}",
            "core/round_index": int(round_index),
            "core/campaign_slug": cfg.campaign.slug,
            "core/labels_as_of_round": int(round_index),
            "core/plugins/transforms_y/name": transform_name,
        },
        registry=registry,
    )
    return round_ctx.for_plugin(category="transform_y", name=transform_name, plugin=get_transform_y(transform_name))
