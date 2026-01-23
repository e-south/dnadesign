# ABOUTME: Builds the round context and registry view for OPAL runs.
# ABOUTME: Centralizes run_id creation and RoundCtx initialization.
"""
Round context helpers.
"""

from __future__ import annotations

from typing import Tuple

from ...config.types import RootConfig
from ...core.round_context import PluginRegistryView, RoundCtx
from ...core.utils import now_iso


def build_round_ctx(
    *,
    cfg: RootConfig,
    as_of_round: int,
    y_dim: int,
    n_train: int,
) -> Tuple[str, PluginRegistryView, RoundCtx]:
    run_id = f"r{int(as_of_round)}-{now_iso()}"
    reg = PluginRegistryView(
        model=cfg.model.name,
        objective=cfg.objective.objective.name,
        selection=cfg.selection.selection.name,
        transform_x=cfg.data.transforms_x.name,
        transform_y=cfg.data.transforms_y.name,
    )
    rctx = RoundCtx(
        core={
            "core/run_id": run_id,
            "core/round_index": int(as_of_round),
            "core/campaign_slug": cfg.campaign.slug,
            "core/labels_as_of_round": int(as_of_round),
            "core/plugins/transforms_x/name": reg.transform_x,
            "core/plugins/transforms_y/name": reg.transform_y,
            "core/plugins/model/name": reg.model,
            "core/plugins/objective/name": reg.objective,
            "core/plugins/selection/name": reg.selection,
            "core/data/y_dim": int(y_dim),
            "core/data/n_train": int(n_train),
        },
        registry=reg,
    )
    return run_id, reg, rctx
