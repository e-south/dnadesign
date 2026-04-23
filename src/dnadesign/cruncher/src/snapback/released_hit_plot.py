"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/released_hit_plot.py

Truthful origin-anchored plotting for released-product snapback solve hits.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dnadesign.cruncher.snapback import released_plot_common as plot_common
from dnadesign.cruncher.snapback.released_plot_context import build_released_hit_plot_model
from dnadesign.cruncher.snapback.released_plot_foldback import render_foldback_panel
from dnadesign.cruncher.snapback.released_plot_precursor import render_precursor_panel
from dnadesign.cruncher.snapback.released_plot_released import render_released_panel
from dnadesign.cruncher.snapback.released_search_models import ReleasedTargetSearchHit

_FIGURE_FACE = plot_common._FIGURE_FACE
_ROW_BOTTOM_Y = plot_common._ROW_BOTTOM_Y
_ROW_TOP_Y = plot_common._ROW_TOP_Y
_SITE_FOOTPRINT_VERTICAL_PAD = plot_common._SITE_FOOTPRINT_VERTICAL_PAD
_site_footprint_bounds = plot_common.site_footprint_bounds


def build_released_hit_plot_context(hit: ReleasedTargetSearchHit) -> dict[str, Any]:
    return build_released_hit_plot_model(hit).model_dump(mode="json")


def render_released_hit_plot(hit: ReleasedTargetSearchHit, output_path: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    context_model = build_released_hit_plot_model(hit)
    context = context_model.model_dump(mode="json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    precursor_width = max(len(hit.precursor_top_strand), 12)
    released_width = max(
        len(context_model.released_product.top_row.sequence),
        len(context_model.released_product.bottom_row.sequence),
        12,
    )
    foldback_width = max(
        len(context_model.foldback.top_row.sequence),
        len(context_model.foldback.bottom_row.sequence),
        8,
    )
    width_ratios = [precursor_width, released_width, foldback_width]
    figure_width = max(15.0, sum(width_ratios) * 0.33)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(figure_width, 4.4),
        dpi=170,
        gridspec_kw={"width_ratios": width_ratios},
    )
    fig.patch.set_facecolor(_FIGURE_FACE)
    render_precursor_panel(
        axes[0],
        context=context_model.precursor,
        nickase_variant_id=context_model.nickase_variant_id,
        release_variant_id=context_model.release_variant_id,
    )
    render_released_panel(axes[1], context=context_model.released_product)
    render_foldback_panel(axes[2], context=context_model.foldback)
    fig.tight_layout(pad=0.32, w_pad=0.34)
    fig.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return context


__all__ = ["build_released_hit_plot_context", "render_released_hit_plot"]
