"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/yiu_workflow/render.py

Render payload-centric YIU bundles from specs.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.app.yiu_workflow.staging import (
    create_bundle_staging_dir,
    promote_staged_bundle,
    remove_managed_path,
)
from dnadesign.cruncher.yiu.bundle_paths import resolve_published_plot_path
from dnadesign.cruncher.yiu.bundle_surface import YiuRenderOutcome, load_render_outcome
from dnadesign.cruncher.yiu.load import load_yiu_spec
from dnadesign.cruncher.yiu.models.bundle import YiuValidationReport, build_validation_report
from dnadesign.cruncher.yiu.normalize import normalize_payload
from dnadesign.cruncher.yiu.publish import publish_payload_bundle
from dnadesign.cruncher.yiu.render import render_bundle_views


def render_yiu_spec(
    path: str | Path,
    *,
    force_overwrite: bool = False,
    emit_renders: bool = False,
) -> tuple[Path, YiuValidationReport]:
    spec, _resolved_spec_path, workspace_root = load_yiu_spec(path)
    normalized = normalize_payload(spec, workspace_root=workspace_root)
    bundle_dir = (workspace_root / spec.output.bundle_dir).resolve()
    bundle_preexisted = bundle_dir.exists()
    if bundle_preexisted and not force_overwrite:
        raise ValueError(f"YIU bundle directory already exists: {bundle_dir}. Use --force-overwrite to replace it.")
    staged_bundle_dir = create_bundle_staging_dir(bundle_dir)
    try:
        publish_payload_bundle(spec=spec, normalized=normalized, bundle_dir=staged_bundle_dir)
        promote_staged_bundle(
            staged_bundle_dir=staged_bundle_dir,
            bundle_dir=bundle_dir,
            force_overwrite=force_overwrite,
        )
    except Exception:
        remove_managed_path(staged_bundle_dir)
        raise
    if bundle_preexisted and not emit_renders:
        remove_managed_path(
            resolve_published_plot_path(
                bundle_dir,
                None if spec.output.published_plot_path is None else str(spec.output.published_plot_path),
            )
        )
    if emit_renders:
        render_bundle_views(bundle_dir)
    report = build_validation_report(
        spec_name=spec.yiu.name,
        normalized=normalized,
        bundle_dir=str(bundle_dir.resolve()),
    )
    return bundle_dir.resolve(), report


def render_yiu_spec_outcome(
    path: str | Path,
    *,
    force_overwrite: bool = False,
    emit_renders: bool = False,
) -> YiuRenderOutcome:
    bundle_dir, report = render_yiu_spec(
        path,
        force_overwrite=force_overwrite,
        emit_renders=emit_renders,
    )
    return load_render_outcome(bundle_dir, report=report)
