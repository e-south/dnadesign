"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/bundle_surface.py

Shared bundle-artifact surfaces for YIU app and CLI boundaries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from dnadesign.cruncher.config.schema_v3 import StrictBaseModel
from dnadesign.cruncher.yiu.bundle_models import PayloadVisualInventory, YiuValidationReport
from dnadesign.cruncher.yiu.bundle_paths import (
    resolve_composite_render_artifact_path,
    resolve_outputs_root,
    resolve_published_plot_path,
)
from dnadesign.cruncher.yiu.bundle_state import load_bundle_state, resolve_bundle_state_paths
from dnadesign.cruncher.yiu.bundle_summary import YiuBundleSummary
from dnadesign.cruncher.yiu.domain_models import (
    JunctionSelection,
    MismatchSelection,
    NormalizedMotifContext,
    OptimizationDecision,
)


class YiuBundleArtifactSurface(StrictBaseModel):
    bundle_dir: str
    outputs_root: str | None = None
    composite_render_artifact_path: str | None = None
    published_plot_artifact_path: str | None = None
    bundle_summary_path: str
    bundle_manifest_path: str
    normalized_payload_path: str
    visual_inventory_path: str


class YiuRenderOutcome(YiuBundleArtifactSurface):
    report: YiuValidationReport


class YiuBundleIntegrity(StrictBaseModel):
    status: Literal["ok"] = "ok"
    checks: list[str] = Field(default_factory=list)
    available_render_count: int = Field(ge=0)


class YiuBundleIntegrityState(StrictBaseModel):
    checks: list[str] = Field(default_factory=list)
    available_renders: list[str] = Field(default_factory=list)
    published_plot_path: str | None = None
    payload_view: dict[str, object] = Field(default_factory=dict)
    split_rows: list[dict[str, object]] = Field(default_factory=list)


class YiuSplitRowDebug(StrictBaseModel):
    fragment_side: str | None = None
    panel_order: int | None = None
    selected_sticky_end_sequence_5to3: str | None = None
    canonical_sticky_end_sequence_5to3: str | None = None
    payload_body_sequence_5to3: str | None = None
    display_payload_body_sequence_5to3: str | None = None
    retained_primary_sequence_5to3: str | None = None
    retained_complement_sequence_3to5: str | None = None
    sticky_end_display_span: dict[str, object] | None = None
    payload_body_display_span: dict[str, object] | None = None
    payload_junction_window: dict[str, object] | None = None
    ghost_excised_context: dict[str, object] | None = None


class YiuShowOutcome(YiuBundleArtifactSurface):
    bundle_contract: Literal["split_yiu_payload_bundle_v4"] = "split_yiu_payload_bundle_v4"
    bundle_summary: YiuBundleSummary
    provenance: dict[str, object] = Field(default_factory=dict)
    payload_label: str | None = None
    input_kind: Literal["user_sequence", "sample_hit"]
    payload_length: int = Field(ge=1)
    selected_payload_sequence: str
    selected_complement_sequence: str
    junction: JunctionSelection
    mismatches: list[MismatchSelection] = Field(default_factory=list)
    pwm_mode: Literal["none", "use_if_available", "require"]
    pwm_effective: bool
    worst_loss: float = Field(ge=0.0)
    total_loss: float = Field(ge=0.0)
    view_ids: list[str] = Field(default_factory=list)
    render_status: Literal["not_requested", "rendered", "missing", "partial", "failed"]
    available_renders: list[str] = Field(default_factory=list)
    integrity: YiuBundleIntegrity
    optimization_decision: OptimizationDecision
    motif_context: NormalizedMotifContext
    split_row_debug: list[YiuSplitRowDebug] = Field(default_factory=list)


def load_payload_visual_inventory(bundle_dir: str | Path) -> PayloadVisualInventory:
    return load_bundle_state(bundle_dir).inventory


def resolve_bundle_artifact_surface(
    bundle_dir: str | Path,
    *,
    inventory: PayloadVisualInventory,
    published_plot_path: str | Path | None = None,
) -> YiuBundleArtifactSurface:
    paths = resolve_bundle_state_paths(bundle_dir)
    resolved = paths.bundle_dir
    outputs_root = resolve_outputs_root(paths.bundle_dir)
    composite_render_artifact_path = resolve_composite_render_artifact_path(paths.bundle_dir, inventory)
    resolved_published_plot = None if published_plot_path is None else Path(published_plot_path).expanduser().resolve()
    if resolved_published_plot is None:
        resolved_published_plot = resolve_published_plot_path(paths.bundle_dir, inventory.published_plot_artifact_path)
    return YiuBundleArtifactSurface(
        bundle_dir=str(resolved),
        outputs_root=None if outputs_root is None else str(outputs_root.resolve()),
        composite_render_artifact_path=(
            None if composite_render_artifact_path is None else str(composite_render_artifact_path.resolve())
        ),
        published_plot_artifact_path=(
            None if resolved_published_plot is None else str(resolved_published_plot.resolve())
        ),
        bundle_summary_path=str(paths.bundle_summary_path.resolve()),
        bundle_manifest_path=str(paths.manifest_path.resolve()),
        normalized_payload_path=str(paths.normalized_payload_path.resolve()),
        visual_inventory_path=str(paths.inventory_path.resolve()),
    )


def build_render_outcome(
    bundle_dir: str | Path,
    *,
    report: YiuValidationReport,
    inventory: PayloadVisualInventory,
) -> YiuRenderOutcome:
    return YiuRenderOutcome(
        report=report,
        **resolve_bundle_artifact_surface(bundle_dir, inventory=inventory).model_dump(mode="json"),
    )


def load_render_outcome(bundle_dir: str | Path, *, report: YiuValidationReport) -> YiuRenderOutcome:
    inventory = load_payload_visual_inventory(bundle_dir)
    return build_render_outcome(bundle_dir, report=report, inventory=inventory)


__all__ = [
    "YiuBundleArtifactSurface",
    "YiuBundleIntegrity",
    "YiuBundleIntegrityState",
    "YiuRenderOutcome",
    "YiuShowOutcome",
    "YiuSplitRowDebug",
    "build_render_outcome",
    "load_payload_visual_inventory",
    "load_render_outcome",
    "resolve_bundle_artifact_surface",
]
