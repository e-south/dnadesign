"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/render.py

Run published BaseRender jobs for payload-centric YIU bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path

from dnadesign.cruncher.yiu.errors import YIU_RENDER_FAILED, YiuContractError, raise_yiu_error
from dnadesign.cruncher.yiu.render_execution import (
    publish_render_outputs as _publish_render_outputs,
)
from dnadesign.cruncher.yiu.render_execution import (
    render_view_panels as _render_view_panels,
)
from dnadesign.cruncher.yiu.render_panels import (
    figure_to_rgba_array,
    load_view_records,
    render_view_panel,
    save_composite_render,
)
from dnadesign.cruncher.yiu.render_plan import (
    YiuRenderProgress,
    load_baserender_runtime,
    prepare_render_plan,
)
from dnadesign.cruncher.yiu.render_state import (
    cleanup_partial_render_outputs,
    persist_failed_render_state,
    persist_render_state,
    render_status,
)


def render_bundle_views(bundle_dir: str | Path) -> dict[str, object]:
    plan = prepare_render_plan(bundle_dir)
    progress = YiuRenderProgress()
    try:
        baserender = load_baserender_runtime()
        _render_view_panels(
            plan=plan,
            baserender_module=baserender,
            progress=progress,
            load_view_records_func=load_view_records,
            render_view_panel_func=render_view_panel,
            figure_to_rgba_array_func=figure_to_rgba_array,
        )
        _publish_render_outputs(
            plan=plan,
            progress=progress,
            save_composite_render_func=save_composite_render,
            copyfile_func=shutil.copyfile,
        )
    except Exception as exc:
        cleanup_partial_render_outputs(plan)
        persist_failed_render_state(plan, progress)
        if isinstance(exc, YiuContractError):
            raise
        raise_yiu_error(YIU_RENDER_FAILED, str(exc))
    updated_state = persist_render_state(
        bundle_state=plan.bundle_state,
        current_render_status=render_status(
            job_count=len(plan.bundle_state.inventory.views),
            rendered_count=progress.rendered_count,
        ),
        rendered_count=progress.rendered_count,
        last_rendered_at=progress.last_rendered_at,
        views=progress.updated_views,
    )
    return {
        "bundle_dir": str(plan.bundle_state.bundle_dir),
        "render_status": updated_state.inventory.render_status,
        "render_count": progress.rendered_count,
        "composite_render_artifact_path": str(plan.composite_render_path),
        "published_plot_artifact_path": None if plan.published_plot_path is None else str(plan.published_plot_path),
        "render_artifact_paths": progress.render_paths,
    }
