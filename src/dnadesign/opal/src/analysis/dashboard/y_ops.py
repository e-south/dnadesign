"""Y-ops helpers used by dashboard scoring and overlays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ...core.round_context import PluginRegistryView, RoundCtx
from ...registries.transforms_y import run_y_ops_pipeline
from .datasets import CampaignInfo


@dataclass(frozen=True)
class YOpEntry:
    name: str
    params: dict


def normalize_y_ops_config(y_ops: Sequence[Mapping[str, Any]]) -> list[YOpEntry]:
    out: list[YOpEntry] = []
    for entry in y_ops or []:
        if not isinstance(entry, Mapping):
            continue
        name = entry.get("name")
        if not name:
            continue
        params = dict(entry.get("params") or {})
        out.append(YOpEntry(name=str(name), params=params))
    return out


def build_round_ctx_for_notebook(
    *,
    info: CampaignInfo,
    run_id: str,
    round_index: int,
    y_dim: int,
    n_train: int,
) -> RoundCtx:
    registry = PluginRegistryView(
        model=info.model_name,
        objective=info.objective_name,
        selection=info.selection_name,
        transform_x="unknown",
        transform_y="unknown",
    )
    ctx = RoundCtx(
        core={
            "core/run_id": str(run_id),
            "core/round_index": int(round_index),
            "core/campaign_slug": info.slug,
            "core/labels_as_of_round": int(round_index),
            "core/plugins/transforms_x/name": registry.transform_x,
            "core/plugins/transforms_y/name": registry.transform_y,
            "core/plugins/model/name": registry.model,
            "core/plugins/objective/name": registry.objective,
            "core/plugins/selection/name": registry.selection,
            "core/data/y_dim": int(y_dim),
            "core/data/n_train": int(n_train),
        },
        registry=registry,
    )
    return ctx


def apply_y_ops_fit_transform(
    *,
    y_ops: Sequence[Mapping[str, Any]],
    y: np.ndarray,
    ctx: RoundCtx,
) -> np.ndarray:
    entries = normalize_y_ops_config(y_ops)
    return run_y_ops_pipeline(stage="fit_transform", y_ops=entries, Y=y, ctx=ctx)


def apply_y_ops_inverse(
    *,
    y_ops: Sequence[Mapping[str, Any]],
    y: np.ndarray,
    ctx: RoundCtx,
) -> np.ndarray:
    entries = normalize_y_ops_config(y_ops)
    return run_y_ops_pipeline(stage="inverse", y_ops=entries, Y=y, ctx=ctx)
