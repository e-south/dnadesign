"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/yiu_workflow/show.py

Inspect payload-centric YIU bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.yiu.bundle_models import payload_summary_dump
from dnadesign.cruncher.yiu.bundle_state import load_bundle_state
from dnadesign.cruncher.yiu.bundle_summary import build_bundle_summary
from dnadesign.cruncher.yiu.bundle_surface import (
    YiuBundleIntegrity,
    YiuShowOutcome,
    YiuSplitRowDebug,
    resolve_bundle_artifact_surface,
)
from dnadesign.cruncher.yiu.errors import YIU_BUNDLE_INVALID, raise_yiu_error
from dnadesign.cruncher.yiu.integrity import validate_bundle_state


def _split_row_debug(rows: list[dict[str, object]]) -> list[YiuSplitRowDebug]:
    debug_rows: list[YiuSplitRowDebug] = []
    for row in rows:
        meta = row.get("meta", {})
        if not isinstance(meta, dict):
            continue
        debug_rows.append(
            YiuSplitRowDebug(
                fragment_side=meta.get("fragment_side"),
                panel_order=meta.get("panel_order"),
                selected_sticky_end_sequence_5to3=meta.get("selected_sticky_end_sequence_5to3"),
                canonical_sticky_end_sequence_5to3=meta.get("canonical_sticky_end_sequence_5to3"),
                payload_body_sequence_5to3=meta.get("payload_body_sequence_5to3"),
                display_payload_body_sequence_5to3=meta.get("display_payload_body_sequence_5to3"),
                retained_primary_sequence_5to3=meta.get("retained_primary_sequence_5to3"),
                retained_complement_sequence_3to5=meta.get("retained_complement_sequence_3to5"),
                sticky_end_display_span=meta.get("sticky_end_display_span"),
                payload_body_display_span=meta.get("payload_body_display_span"),
                payload_junction_window=meta.get("payload_junction_window"),
                ghost_excised_context=meta.get("ghost_excised_context"),
            )
        )
    return debug_rows


def show_yiu_bundle(bundle_dir: str | Path, *, verbose: bool = False) -> YiuShowOutcome:
    resolved = Path(bundle_dir).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"YIU bundle directory not found: {resolved}")
    state = load_bundle_state(resolved, include_normalized=True)
    if state.normalized is None:
        raise_yiu_error(YIU_BUNDLE_INVALID, "normalized_payload.json is required for YIU bundle inspection")
    integrity = validate_bundle_state(
        bundle_dir=state.bundle_dir,
        manifest=state.manifest,
        inventory=state.inventory,
        normalized=state.normalized,
    )
    artifact_surface = resolve_bundle_artifact_surface(
        state.bundle_dir,
        inventory=state.inventory,
        published_plot_path=integrity.published_plot_path,
    )
    outcome_kwargs = {
        "bundle_contract": state.manifest.bundle_contract,
        "bundle_summary": build_bundle_summary(normalized=state.normalized, inventory=state.inventory),
        "provenance": state.manifest.provenance,
        **payload_summary_dump(state.manifest),
        "view_ids": [view.view_id for view in state.inventory.views],
        "render_status": state.inventory.render_status,
        "available_renders": integrity.available_renders,
        "integrity": YiuBundleIntegrity(
            status="ok",
            checks=integrity.checks,
            available_render_count=len(integrity.available_renders),
        ),
        "optimization_decision": state.normalized.optimization_decision,
        "motif_context": state.normalized.motif_context,
    }
    if verbose:
        outcome_kwargs["split_row_debug"] = _split_row_debug(integrity.split_rows)
    return YiuShowOutcome(
        **artifact_surface.model_dump(mode="json"),
        **outcome_kwargs,
    )
