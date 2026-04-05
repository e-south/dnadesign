"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/render_execution.py

Panel execution and output publication helpers for payload-centric YIU bundles.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from dnadesign.cruncher.yiu.errors import YIU_RENDER_FAILED, raise_yiu_error
from dnadesign.cruncher.yiu.render_plan import BaserenderRuntime, YiuRenderPlan, YiuRenderProgress


def _requested_view(view):
    return view.model_copy(update={"render_requested": True})


def _completed_view(view, *, last_rendered_at: str):
    return view.model_copy(
        update={"render_requested": True, "render_completed": True, "last_rendered_at": last_rendered_at}
    )


def _close_panel(panel) -> None:
    try:
        import matplotlib.pyplot as plt

        plt.close(panel)
    except TypeError:
        pass


def render_view_panels(
    *,
    plan: YiuRenderPlan,
    baserender_module: BaserenderRuntime,
    progress: YiuRenderProgress,
    load_view_records_func: Callable[..., object],
    render_view_panel_func: Callable[..., object],
    figure_to_rgba_array_func: Callable[[object], object],
) -> None:
    for view in plan.bundle_state.inventory.views:
        requested_view = _requested_view(view)
        contract_path = (plan.bundle_state.bundle_dir / view.view_contract_path).resolve()
        try:
            records = load_view_records_func(contract_path, view=view, baserender_module=baserender_module)
            panel = render_view_panel_func(
                baserender_module=baserender_module,
                records=records,
                renderer_kind=view.renderer_kind,
                style_preset=view.style_preset,
                style_overrides=view.style_overrides,
            )
        except Exception as exc:
            progress.updated_views.append(requested_view)
            raise_yiu_error(YIU_RENDER_FAILED, f"BaseRender failed for view {view.view_id!r} ({exc})")
        try:
            progress.panel_images.append(figure_to_rgba_array_func(panel))
        finally:
            _close_panel(panel)
        progress.last_rendered_at = datetime.now(timezone.utc).isoformat()
        progress.rendered_count += 1
        progress.updated_views.append(_completed_view(view, last_rendered_at=progress.last_rendered_at))


def publish_render_outputs(
    *,
    plan: YiuRenderPlan,
    progress: YiuRenderProgress,
    save_composite_render_func: Callable[..., None],
    copyfile_func: Callable[[Path, Path], object],
) -> None:
    try:
        save_composite_render_func(panel_images=progress.panel_images, render_path=plan.composite_render_path)
    except Exception as exc:
        raise_yiu_error(YIU_RENDER_FAILED, f"BaseRender composite assembly failed ({exc})")
    if not plan.composite_render_path.exists():
        raise_yiu_error(YIU_RENDER_FAILED, "YIU composite render did not create payload_views.pdf")
    progress.render_paths.append(str(plan.composite_render_path))
    published_plot_path = plan.published_plot_path
    if published_plot_path is None:
        return
    if published_plot_path != plan.composite_render_path:
        try:
            published_plot_path.parent.mkdir(parents=True, exist_ok=True)
            copyfile_func(plan.composite_render_path, published_plot_path)
        except Exception as exc:
            raise_yiu_error(YIU_RENDER_FAILED, f"BaseRender published-plot mirror failed ({exc})")
    published_plot_resolved = str(published_plot_path)
    if published_plot_resolved not in progress.render_paths:
        progress.render_paths.append(published_plot_resolved)


__all__ = ["publish_render_outputs", "render_view_panels"]
