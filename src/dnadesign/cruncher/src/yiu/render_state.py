"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/yiu/render_state.py

Render-state persistence and artifact cleanup helpers for payload-centric YIU.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.yiu.bundle_models import PayloadViewEntry
from dnadesign.cruncher.yiu.bundle_state import RenderStatus, YiuBundleState
from dnadesign.cruncher.yiu.bundle_summary import build_bundle_summary
from dnadesign.cruncher.yiu.render_plan import YiuRenderPlan, YiuRenderProgress
from dnadesign.cruncher.yiu.view_io import write_json_payload


def render_status(*, job_count: int, rendered_count: int) -> RenderStatus:
    if job_count <= 0:
        return "not_requested"
    if rendered_count <= 0:
        return "missing"
    if rendered_count >= job_count:
        return "rendered"
    return "partial"


def persist_render_state(
    *,
    bundle_state: YiuBundleState,
    current_render_status: RenderStatus,
    rendered_count: int,
    last_rendered_at: str | None,
    views: list[PayloadViewEntry],
) -> YiuBundleState:
    updated_state = bundle_state.with_render_state(
        rendered_count=rendered_count,
        last_rendered_at=last_rendered_at,
        views=views,
        render_status=current_render_status,
    )
    updated_state.persist()
    if updated_state.normalized is not None:
        write_json_payload(
            updated_state.paths.bundle_summary_path,
            build_bundle_summary(
                normalized=updated_state.normalized,
                inventory=updated_state.inventory,
            ).model_dump(mode="json"),
        )
    return updated_state


def _views_with_progress(plan: YiuRenderPlan, progress: YiuRenderProgress) -> list[PayloadViewEntry]:
    original_views = plan.bundle_state.inventory.views
    return progress.updated_views + original_views[len(progress.updated_views) :]


def persist_failed_render_state(plan: YiuRenderPlan, progress: YiuRenderProgress) -> None:
    persist_render_state(
        bundle_state=plan.bundle_state,
        current_render_status="failed",
        rendered_count=progress.rendered_count,
        last_rendered_at=progress.last_rendered_at,
        views=_views_with_progress(plan, progress),
    )


def cleanup_partial_render_outputs(plan: YiuRenderPlan) -> None:
    seen_paths: set[Path] = set()
    for path in (plan.composite_render_path, plan.published_plot_path):
        if path is None:
            continue
        resolved = path.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        if resolved.exists():
            resolved.unlink()


__all__ = [
    "cleanup_partial_render_outputs",
    "persist_failed_render_state",
    "persist_render_state",
    "render_status",
]
