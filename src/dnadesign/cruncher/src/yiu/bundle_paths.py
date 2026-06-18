"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/yiu/bundle_paths.py

Shared bundle-path resolution for YIU publication, rendering, and inspection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.yiu.bundle_models import PayloadVisualInventory


def resolve_outputs_root(bundle_dir: Path) -> Path | None:
    resolved = bundle_dir.resolve()
    for candidate in (resolved, *resolved.parents):
        if candidate.name == "outputs":
            return candidate
    return None


def resolve_workspace_root(bundle_dir: Path) -> Path | None:
    resolved = bundle_dir.resolve()
    for candidate in (resolved, *resolved.parents):
        if candidate.name == "outputs":
            return candidate.parent
        if candidate.name == "bundles":
            return candidate.parent
    return None


def resolve_published_plot_path(bundle_dir: Path, relative_path: str | None) -> Path | None:
    if relative_path is None:
        return None
    workspace_root = resolve_workspace_root(bundle_dir)
    if workspace_root is None:
        return None
    return (workspace_root / relative_path).resolve()


def resolve_composite_render_artifact_path(
    bundle_dir: Path,
    inventory: PayloadVisualInventory,
) -> Path | None:
    resolved_bundle_dir = bundle_dir.resolve()
    expected_path = (
        (resolved_bundle_dir / inventory.composite_render_artifact_path).resolve()
        if inventory.composite_render_artifact_path is not None
        else None
    )
    for view in inventory.views:
        render_path = (resolved_bundle_dir / view.render_artifact_path).resolve()
        if expected_path is not None and render_path != expected_path:
            raise ValueError("published view render paths diverge from the bundle composite render target")
        expected_path = render_path if expected_path is None else expected_path
        if expected_path != render_path:
            raise ValueError("published view render paths diverge from the bundle composite render target")
    return expected_path


def resolve_expected_render_artifact_paths(
    bundle_dir: Path,
    inventory: PayloadVisualInventory,
) -> list[Path]:
    composite_path = resolve_composite_render_artifact_path(bundle_dir, inventory)
    if composite_path is not None:
        return [composite_path]
    resolved_bundle_dir = bundle_dir.resolve()
    return sorted({(resolved_bundle_dir / view.render_artifact_path).resolve() for view in inventory.views}, key=str)


__all__ = [
    "resolve_composite_render_artifact_path",
    "resolve_expected_render_artifact_paths",
    "resolve_outputs_root",
    "resolve_published_plot_path",
    "resolve_workspace_root",
]
