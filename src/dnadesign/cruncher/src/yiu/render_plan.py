"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/yiu/render_plan.py

Render preflight, runtime loading, and plan models for payload-centric YIU.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from dnadesign.cruncher.viz.mpl import (
    ensure_mpl_cache,
    ensure_workspace_mpl_cache,
    infer_workspace_root_from_output_artifact,
)
from dnadesign.cruncher.yiu.bundle_models import PayloadViewEntry
from dnadesign.cruncher.yiu.bundle_paths import (
    resolve_composite_render_artifact_path,
    resolve_published_plot_path,
)
from dnadesign.cruncher.yiu.bundle_state import YiuBundleState, load_bundle_state
from dnadesign.cruncher.yiu.errors import YIU_BUNDLE_INVALID, YIU_RENDER_FAILED, raise_yiu_error
from dnadesign.cruncher.yiu.integrity import validate_bundle_state


class BaserenderRuntime(Protocol):
    def adapt_records(
        self,
        rows,
        *,
        adapter_kind: str,
        alphabet: str,
    ): ...

    def render(
        self,
        record_or_records,
        *,
        renderer: str,
        style: dict[str, object],
        grid: dict[str, int] | None = None,
    ): ...


@dataclass(frozen=True)
class YiuRenderPlan:
    bundle_state: YiuBundleState
    composite_render_path: Path
    published_plot_path: Path | None = None


@dataclass
class YiuRenderProgress:
    rendered_count: int = 0
    last_rendered_at: str | None = None
    updated_views: list[PayloadViewEntry] = field(default_factory=list)
    panel_images: list[Any] = field(default_factory=list)
    render_paths: list[str] = field(default_factory=list)


def load_baserender_runtime() -> BaserenderRuntime:
    return importlib.import_module("dnadesign.baserender")


def _configure_render_runtime(bundle_state: YiuBundleState) -> None:
    workspace_root = infer_workspace_root_from_output_artifact(bundle_state.paths.inventory_path)
    if workspace_root is not None:
        ensure_workspace_mpl_cache(workspace_root)
        return
    ensure_mpl_cache(bundle_state.bundle_dir)


def prepare_render_plan(bundle_dir: str | Path) -> YiuRenderPlan:
    bundle_state = load_bundle_state(bundle_dir, include_normalized=True)
    if bundle_state.normalized is None:
        raise_yiu_error(YIU_BUNDLE_INVALID, "normalized_payload.json is required for YIU render preflight")
    validate_bundle_state(
        bundle_dir=bundle_state.bundle_dir,
        manifest=bundle_state.manifest,
        inventory=bundle_state.inventory,
        normalized=bundle_state.normalized,
    )
    try:
        composite_render_path = resolve_composite_render_artifact_path(bundle_state.bundle_dir, bundle_state.inventory)
    except ValueError as exc:
        raise_yiu_error(YIU_RENDER_FAILED, str(exc))
    if composite_render_path is None:
        raise_yiu_error(YIU_RENDER_FAILED, "YIU render inventory is empty")
    _configure_render_runtime(bundle_state)
    published_plot_path = resolve_published_plot_path(
        bundle_state.bundle_dir,
        bundle_state.inventory.published_plot_artifact_path,
    )
    if bundle_state.inventory.published_plot_artifact_path is not None and published_plot_path is None:
        raise_yiu_error(YIU_RENDER_FAILED, "YIU published plot path is set but the workspace root cannot be resolved")
    return YiuRenderPlan(
        bundle_state=bundle_state,
        composite_render_path=composite_render_path.resolve(),
        published_plot_path=None if published_plot_path is None else published_plot_path.resolve(),
    )


__all__ = [
    "BaserenderRuntime",
    "YiuRenderPlan",
    "YiuRenderProgress",
    "load_baserender_runtime",
    "prepare_render_plan",
]
