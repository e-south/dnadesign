"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/render.py

Run published BaseRender jobs for payload-centric YIU bundles.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

from dnadesign.cruncher.viz.mpl import (
    ensure_mpl_cache,
    ensure_workspace_mpl_cache,
    infer_workspace_root_from_output_artifact,
)
from dnadesign.cruncher.yiu.bundle_paths import (
    resolve_composite_render_artifact_path,
    resolve_published_plot_path,
)
from dnadesign.cruncher.yiu.bundle_state import RenderStatus, YiuBundleState, load_bundle_state
from dnadesign.cruncher.yiu.errors import YIU_BUNDLE_INVALID, YIU_RENDER_FAILED, raise_yiu_error
from dnadesign.cruncher.yiu.integrity import validate_bundle_state
from dnadesign.cruncher.yiu.models.bundle import PayloadViewEntry
from dnadesign.cruncher.yiu.render_panels import (
    figure_to_rgba_array,
    load_view_records,
    render_view_panel,
    save_composite_render,
)


def _render_status(*, job_count: int, rendered_count: int) -> RenderStatus:
    if job_count <= 0:
        return "not_requested"
    if rendered_count <= 0:
        return "missing"
    if rendered_count >= job_count:
        return "rendered"
    return "partial"


def _persist_render_state(
    *,
    bundle_state: YiuBundleState,
    render_status: RenderStatus,
    rendered_count: int,
    last_rendered_at: str | None,
    views: list[PayloadViewEntry],
) -> YiuBundleState:
    updated_state = bundle_state.with_render_state(
        rendered_count=rendered_count,
        last_rendered_at=last_rendered_at,
        views=views,
        render_status=render_status,
    )
    updated_state.persist()
    return updated_state


def render_bundle_views(bundle_dir: str | Path) -> dict[str, object]:
    bundle_state = load_bundle_state(bundle_dir, include_normalized=True)
    if bundle_state.normalized is None:
        raise_yiu_error(YIU_BUNDLE_INVALID, "normalized_payload.json is required for YIU render preflight")
    validate_bundle_state(
        bundle_dir=bundle_state.bundle_dir,
        manifest=bundle_state.manifest,
        inventory=bundle_state.inventory,
        normalized=bundle_state.normalized,
    )
    resolved = bundle_state.bundle_dir
    inventory = bundle_state.inventory
    workspace_root = infer_workspace_root_from_output_artifact(bundle_state.paths.inventory_path)
    if workspace_root is not None:
        ensure_workspace_mpl_cache(workspace_root)
    else:
        ensure_mpl_cache(resolved)

    baserender = importlib.import_module("dnadesign.baserender")
    rendered_count = 0
    render_paths: list[str] = []
    last_rendered_at: str | None = None
    updated_views: list[PayloadViewEntry] = []
    panel_images = []
    try:
        composite_render_path = resolve_composite_render_artifact_path(resolved, inventory)
    except ValueError as exc:
        raise_yiu_error(YIU_RENDER_FAILED, str(exc))
    for view in inventory.views:
        contract_path = (resolved / view.view_contract_path).resolve()
        requested_view = view.model_copy(update={"render_requested": True})
        try:
            records = load_view_records(contract_path, view=view, baserender_module=baserender)
            panel = render_view_panel(
                baserender_module=baserender,
                records=records,
                renderer_kind=view.renderer_kind,
                style_preset=view.style_preset,
                style_overrides=view.style_overrides,
            )
        except Exception as exc:
            updated_views.append(requested_view)
            _persist_render_state(
                bundle_state=bundle_state,
                render_status="failed",
                rendered_count=rendered_count,
                last_rendered_at=last_rendered_at,
                views=updated_views + inventory.views[len(updated_views) :],
            )
            raise_yiu_error(YIU_RENDER_FAILED, f"BaseRender failed for view {view.view_id!r} ({exc})")
        try:
            panel_images.append(figure_to_rgba_array(panel))
        finally:
            try:
                import matplotlib.pyplot as plt

                plt.close(panel)
            except TypeError:
                pass
        last_rendered_at = datetime.now(timezone.utc).isoformat()
        rendered_count += 1
        updated_views.append(
            requested_view.model_copy(
                update={"render_requested": True, "render_completed": True, "last_rendered_at": last_rendered_at}
            )
        )

    if composite_render_path is None:
        raise_yiu_error(YIU_RENDER_FAILED, "YIU render inventory is empty")
    try:
        save_composite_render(panel_images=panel_images, render_path=composite_render_path)
    except Exception as exc:
        _persist_render_state(
            bundle_state=bundle_state,
            render_status="failed",
            rendered_count=rendered_count,
            last_rendered_at=last_rendered_at,
            views=updated_views,
        )
        raise_yiu_error(YIU_RENDER_FAILED, f"BaseRender composite assembly failed ({exc})")
    if not composite_render_path.exists():
        raise_yiu_error(YIU_RENDER_FAILED, "YIU composite render did not create payload_views.pdf")
    render_paths.append(str(composite_render_path.resolve()))

    published_plot_path = resolve_published_plot_path(resolved, inventory.published_plot_artifact_path)
    if inventory.published_plot_artifact_path is not None and published_plot_path is None:
        raise_yiu_error(YIU_RENDER_FAILED, "YIU published plot path is set but the workspace root cannot be resolved")
    if published_plot_path is not None:
        if published_plot_path != composite_render_path.resolve():
            published_plot_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(composite_render_path, published_plot_path)
        published_plot_resolved = str(published_plot_path.resolve())
        if published_plot_resolved not in render_paths:
            render_paths.append(published_plot_resolved)

    updated_state = _persist_render_state(
        bundle_state=bundle_state,
        render_status=_render_status(job_count=len(inventory.views), rendered_count=rendered_count),
        rendered_count=rendered_count,
        last_rendered_at=last_rendered_at,
        views=updated_views,
    )
    return {
        "bundle_dir": str(resolved),
        "render_status": updated_state.inventory.render_status,
        "render_count": rendered_count,
        "composite_render_artifact_path": str(composite_render_path.resolve()),
        "published_plot_artifact_path": None if published_plot_path is None else str(published_plot_path.resolve()),
        "render_artifact_paths": render_paths,
    }
