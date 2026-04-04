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
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dnadesign.cruncher.viz.mpl import (
    ensure_mpl_cache,
    ensure_workspace_mpl_cache,
    infer_workspace_root_from_output_artifact,
)
from dnadesign.cruncher.yiu.errors import YIU_RENDER_FAILED, raise_yiu_error
from dnadesign.cruncher.yiu.integrity import resolve_published_plot_path
from dnadesign.cruncher.yiu.models.bundle import PayloadBundleManifest, PayloadViewEntry, PayloadVisualInventory


def _render_status(*, job_count: int, rendered_count: int) -> str:
    if job_count <= 0:
        return "not_requested"
    if rendered_count <= 0:
        return "missing"
    if rendered_count >= job_count:
        return "rendered"
    return "partial"


def _load_bundle_state(bundle_dir: Path) -> tuple[PayloadBundleManifest, PayloadVisualInventory, Path, Path]:
    manifest_path = bundle_dir / "bundle_manifest.json"
    inventory_path = bundle_dir / "visual_inventory.json"
    if not inventory_path.exists():
        raise FileNotFoundError(f"visual inventory not found: {inventory_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"bundle manifest not found: {manifest_path}")
    manifest = PayloadBundleManifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
    inventory = PayloadVisualInventory.model_validate(json.loads(inventory_path.read_text(encoding="utf-8")))
    return manifest, inventory, manifest_path, inventory_path


def _persist_bundle_state(
    *,
    manifest: PayloadBundleManifest,
    inventory: PayloadVisualInventory,
    manifest_path: Path,
    inventory_path: Path,
) -> None:
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2), encoding="utf-8")
    inventory_path.write_text(json.dumps(inventory.model_dump(mode="json"), indent=2), encoding="utf-8")


def _adapter_for_view(view: PayloadViewEntry):
    if view.contract_kind == "sequence_evidence_map_v1":
        from dnadesign.baserender.src.adapters.sequence_evidence_map_v1 import SequenceEvidenceMapV1Adapter

        return SequenceEvidenceMapV1Adapter(columns={}, policies={}, alphabet="IUPAC_DNA")
    if view.contract_kind == "yiu_payload_visual_v1":
        from dnadesign.baserender.src.adapters.yiu_payload_visual_v1 import YiuPayloadVisualV1Adapter

        return YiuPayloadVisualV1Adapter(columns={}, policies={}, alphabet="IUPAC_DNA")
    raise_yiu_error(YIU_RENDER_FAILED, f"unsupported YIU view contract for rendering: {view.contract_kind}")


def _load_contract_rows(contract_path: Path, *, input_kind: str) -> list[dict[str, object]]:
    if input_kind == "jsonl":
        return [json.loads(line) for line in contract_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    if isinstance(payload, dict):
        return [payload]
    raise_yiu_error(YIU_RENDER_FAILED, f"render input must decode to a mapping or list: {contract_path}")


def _load_view_records(contract_path: Path, *, view: PayloadViewEntry):
    adapter = _adapter_for_view(view)
    rows = _load_contract_rows(contract_path, input_kind=view.input_kind)
    return [adapter.apply(row, row_index=index) for index, row in enumerate(rows)]


def _render_view_panel(
    *,
    baserender_module,
    records,
    renderer_kind: str,
    style_preset: str | None,
    style_overrides: dict[str, object],
) -> object:
    record_or_records = records[0] if len(records) == 1 else records
    grid = {"ncols": 2} if len(records) == 2 else None
    return baserender_module.render(
        record_or_records,
        renderer=renderer_kind,
        style={"preset": style_preset, "overrides": style_overrides},
        grid=grid,
    )


def _figure_to_rgba_array(fig) -> np.ndarray:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    return np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((height, width, 4))


def _save_composite_render(*, panel_images: list[np.ndarray], render_path: Path) -> None:
    import matplotlib.pyplot as plt

    if not panel_images:
        raise ValueError("YIU composite render requires at least one panel image")
    height_ratios = [max(1, image.shape[0]) for image in panel_images]
    max_width = max(image.shape[1] for image in panel_images)
    total_height = sum(height_ratios)
    composite = plt.figure(figsize=(max_width / 180.0, total_height / 180.0), dpi=180)
    try:
        axes = composite.subplots(
            nrows=len(panel_images),
            ncols=1,
            gridspec_kw={"height_ratios": height_ratios, "hspace": 0.08},
        )
        if not isinstance(axes, np.ndarray):
            axes = np.asarray([axes], dtype=object)
        for axis, image in zip(axes.tolist(), panel_images, strict=True):
            axis.imshow(image)
            axis.set_axis_off()
        composite.patch.set_facecolor("white")
        composite.patch.set_alpha(1.0)
        render_path.parent.mkdir(parents=True, exist_ok=True)
        composite.savefig(
            render_path,
            format=render_path.suffix.lstrip(".") or "pdf",
            bbox_inches="tight",
            pad_inches=0.02,
            facecolor="white",
        )
    finally:
        plt.close(composite)


def render_bundle_views(bundle_dir: str | Path) -> dict[str, object]:
    resolved = Path(bundle_dir).expanduser().resolve()
    manifest, inventory, manifest_path, inventory_path = _load_bundle_state(resolved)
    workspace_root = infer_workspace_root_from_output_artifact(inventory_path)
    if workspace_root is not None:
        ensure_workspace_mpl_cache(workspace_root)
    else:
        ensure_mpl_cache(resolved)

    baserender = importlib.import_module("dnadesign.baserender")
    rendered_count = 0
    render_paths: list[str] = []
    last_rendered_at: str | None = None
    updated_views: list[PayloadViewEntry] = []
    panel_images: list[np.ndarray] = []
    composite_render_path: Path | None = None
    expected_composite_render_path = (
        (resolved / inventory.composite_render_artifact_path).resolve()
        if inventory.composite_render_artifact_path is not None
        else None
    )
    for view in inventory.views:
        contract_path = (resolved / view.view_contract_path).resolve()
        render_path = (resolved / view.render_artifact_path).resolve()
        if expected_composite_render_path is not None and render_path != expected_composite_render_path:
            raise_yiu_error(
                YIU_RENDER_FAILED,
                "published view render paths diverge from the bundle composite render target",
            )
        composite_render_path = render_path if composite_render_path is None else composite_render_path
        if composite_render_path != render_path:
            raise_yiu_error(
                YIU_RENDER_FAILED,
                "published view render paths diverge from the bundle composite render target",
            )
        requested_view = view.model_copy(update={"render_requested": True})
        try:
            records = _load_view_records(contract_path, view=view)
            panel = _render_view_panel(
                baserender_module=baserender,
                records=records,
                renderer_kind=view.renderer_kind,
                style_preset=view.style_preset,
                style_overrides=view.style_overrides,
            )
        except Exception as exc:
            updated_views.append(requested_view)
            failed_inventory = inventory.model_copy(
                update={
                    "render_count": rendered_count,
                    "render_status": "failed",
                    "last_rendered_at": last_rendered_at,
                    "views": updated_views + inventory.views[len(updated_views) :],
                }
            )
            failed_manifest = manifest.model_copy(
                update={
                    "render_status": "failed",
                    "view_contracts": failed_inventory.views,
                }
            )
            _persist_bundle_state(
                manifest=failed_manifest,
                inventory=failed_inventory,
                manifest_path=manifest_path,
                inventory_path=inventory_path,
            )
            raise_yiu_error(YIU_RENDER_FAILED, f"BaseRender failed for view {view.view_id!r} ({exc})")
        try:
            panel_images.append(_figure_to_rgba_array(panel))
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
        _save_composite_render(panel_images=panel_images, render_path=composite_render_path)
    except Exception as exc:
        failed_inventory = inventory.model_copy(
            update={
                "render_count": rendered_count,
                "render_status": "failed",
                "last_rendered_at": last_rendered_at,
                "views": updated_views,
            }
        )
        failed_manifest = manifest.model_copy(
            update={"render_status": "failed", "view_contracts": failed_inventory.views}
        )
        _persist_bundle_state(
            manifest=failed_manifest,
            inventory=failed_inventory,
            manifest_path=manifest_path,
            inventory_path=inventory_path,
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

    updated_inventory = inventory.model_copy(
        update={
            "render_count": rendered_count,
            "render_status": _render_status(job_count=len(inventory.views), rendered_count=rendered_count),
            "last_rendered_at": last_rendered_at,
            "views": updated_views,
        }
    )
    updated_manifest = manifest.model_copy(
        update={
            "render_status": updated_inventory.render_status,
            "view_contracts": updated_inventory.views,
        }
    )
    _persist_bundle_state(
        manifest=updated_manifest,
        inventory=updated_inventory,
        manifest_path=manifest_path,
        inventory_path=inventory_path,
    )
    return {
        "bundle_dir": str(resolved),
        "render_status": updated_inventory.render_status,
        "render_count": rendered_count,
        "composite_render_artifact_path": str(composite_render_path.resolve()),
        "published_plot_artifact_path": None if published_plot_path is None else str(published_plot_path.resolve()),
        "render_artifact_paths": render_paths,
    }
