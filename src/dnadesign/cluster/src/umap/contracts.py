"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/umap/contracts.py

Typed UMAP runtime request contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_PLOT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "alpha": 0.5,
    "size": 4.0,
    "dims": [12, 12],
    "font_scale": 1.2,
    "legend": {"ncol": 1, "bbox": (1.02, 1.0), "max_items": 40, "frameon": False},
    "highlight": {
        "overlay": True,
        "size_multiplier": 1.6,
        "alpha": 0.95,
        "facecolor": "none",
        "edgecolor": "red",
        "linewidth": 0.9,
        "marker": "o",
        "legend": False,
    },
}


@dataclass(frozen=True, slots=True)
class ResolvedUmapRequest:
    neighbors: int
    min_dist: float
    metric: str
    random_state: int
    render_plots: bool
    color_by: tuple[str, ...]
    highlight_payload: dict[str, Any] | None
    alpha: float
    size: float
    dims: tuple[int, int]
    font_scale: float
    legend: dict[str, Any]
    highlight_style: dict[str, Any]
