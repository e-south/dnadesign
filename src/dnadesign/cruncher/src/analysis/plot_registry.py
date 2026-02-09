"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/plot_registry.py

Defines the curated analysis plot suite for v3.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PlotSpec:
    key: str
    label: str
    requires: Tuple[str, ...]
    outputs: Tuple[str, ...]
    group: str
    description: str


PLOT_SPECS: tuple[PlotSpec, ...] = (
    PlotSpec(
        "opt_trajectory_story",
        "Optimization trajectory (story)",
        ("sequences", "baseline"),
        ("plot__opt_trajectory_story.{ext}",),
        "summary",
        "Story view: baseline density, best-so-far progression, selected top-k, and consensus anchors.",
    ),
    PlotSpec(
        "opt_trajectory_debug",
        "Optimization trajectory (debug)",
        ("sequences", "baseline"),
        ("plot__opt_trajectory_debug.{ext}",),
        "diagnostics",
        "Debug view: chain trajectories and phase boundary markers with corrected cold-chain semantics.",
    ),
    PlotSpec(
        "elites_nn_distance",
        "Elite nearest-neighbor distance",
        ("elites", "baseline_hits"),
        ("plot__elites_nn_distance.{ext}",),
        "summary",
        "Nearest-neighbor distance distribution for elites in TFBS-core space.",
    ),
    PlotSpec(
        "overlap_panel",
        "Overlap panel",
        ("elites",),
        ("plot__overlap_panel.{ext}",),
        "overlap",
        "Overlap heatmap and distribution (or compact summary for large TF sets).",
    ),
    PlotSpec(
        "health_panel",
        "Health panel",
        ("trace",),
        ("plot__health_panel.{ext}",),
        "diagnostics",
        "Swap acceptance and move acceptance summary.",
    ),
)
