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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

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
from dnadesign.cruncher.yiu.errors import YIU_BUNDLE_INVALID, YIU_RENDER_FAILED, YiuContractError, raise_yiu_error
from dnadesign.cruncher.yiu.integrity import validate_bundle_state
from dnadesign.cruncher.yiu.models.bundle import PayloadViewEntry
from dnadesign.cruncher.yiu.render_panels import (
    figure_to_rgba_array,
    load_view_records,
    render_view_panel,
    save_composite_render,
)


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


def load_baserender_runtime() -> BaserenderRuntime:
    return importlib.import_module("dnadesign.baserender")


def _configure_render_runtime(bundle_state: YiuBundleState) -> None:
    workspace_root = infer_workspace_root_from_output_artifact(bundle_state.paths.inventory_path)
    if workspace_root is not None:
        ensure_workspace_mpl_cache(workspace_root)
        return
    ensure_mpl_cache(bundle_state.bundle_dir)


def _prepare_render_plan(bundle_dir: str | Path) -> YiuRenderPlan:
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


def _requested_view(view: PayloadViewEntry) -> PayloadViewEntry:
    return view.model_copy(update={"render_requested": True})


def _completed_view(view: PayloadViewEntry, *, last_rendered_at: str) -> PayloadViewEntry:
    return view.model_copy(
        update={"render_requested": True, "render_completed": True, "last_rendered_at": last_rendered_at}
    )


def _views_with_progress(plan: YiuRenderPlan, progress: YiuRenderProgress) -> list[PayloadViewEntry]:
    original_views = plan.bundle_state.inventory.views
    return progress.updated_views + original_views[len(progress.updated_views) :]


def _persist_failed_render_state(plan: YiuRenderPlan, progress: YiuRenderProgress) -> None:
    _persist_render_state(
        bundle_state=plan.bundle_state,
        render_status="failed",
        rendered_count=progress.rendered_count,
        last_rendered_at=progress.last_rendered_at,
        views=_views_with_progress(plan, progress),
    )


def _cleanup_partial_render_outputs(plan: YiuRenderPlan) -> None:
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


def _render_view_panels(
    *,
    plan: YiuRenderPlan,
    baserender_module: BaserenderRuntime,
    progress: YiuRenderProgress,
) -> None:
    for view in plan.bundle_state.inventory.views:
        requested_view = _requested_view(view)
        contract_path = (plan.bundle_state.bundle_dir / view.view_contract_path).resolve()
        try:
            records = load_view_records(contract_path, view=view, baserender_module=baserender_module)
            panel = render_view_panel(
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
            progress.panel_images.append(figure_to_rgba_array(panel))
        finally:
            try:
                import matplotlib.pyplot as plt

                plt.close(panel)
            except TypeError:
                pass
        progress.last_rendered_at = datetime.now(timezone.utc).isoformat()
        progress.rendered_count += 1
        progress.updated_views.append(_completed_view(view, last_rendered_at=progress.last_rendered_at))


def _publish_render_outputs(plan: YiuRenderPlan, progress: YiuRenderProgress) -> None:
    try:
        save_composite_render(panel_images=progress.panel_images, render_path=plan.composite_render_path)
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
            shutil.copyfile(plan.composite_render_path, published_plot_path)
        except Exception as exc:
            raise_yiu_error(YIU_RENDER_FAILED, f"BaseRender published-plot mirror failed ({exc})")
    published_plot_resolved = str(published_plot_path)
    if published_plot_resolved not in progress.render_paths:
        progress.render_paths.append(published_plot_resolved)


def render_bundle_views(bundle_dir: str | Path) -> dict[str, object]:
    plan = _prepare_render_plan(bundle_dir)
    progress = YiuRenderProgress()
    try:
        baserender = load_baserender_runtime()
        _render_view_panels(plan=plan, baserender_module=baserender, progress=progress)
        _publish_render_outputs(plan, progress)
    except Exception as exc:
        _cleanup_partial_render_outputs(plan)
        _persist_failed_render_state(plan, progress)
        if isinstance(exc, YiuContractError):
            raise
        raise_yiu_error(YIU_RENDER_FAILED, str(exc))
    updated_state = _persist_render_state(
        bundle_state=plan.bundle_state,
        render_status=_render_status(
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
