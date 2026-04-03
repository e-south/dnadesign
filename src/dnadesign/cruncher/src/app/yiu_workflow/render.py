"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/yiu_workflow/render.py

Render payload-centric YIU bundles from specs.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path

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
    if bundle_dir.exists():
        if not force_overwrite:
            raise ValueError(f"YIU bundle directory already exists: {bundle_dir}. Use --force-overwrite to replace it.")
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    publish_payload_bundle(spec=spec, normalized=normalized, bundle_dir=bundle_dir)
    if emit_renders:
        render_bundle_views(bundle_dir)
    report = build_validation_report(
        spec_name=spec.yiu.name,
        normalized=normalized,
        bundle_dir=str(bundle_dir.resolve()),
    )
    return bundle_dir.resolve(), report
