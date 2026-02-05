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
        "run_dashboard",
        "Run dashboard",
        ("sequences", "elites"),
        ("plot__run__dashboard.{ext}",),
        "summary",
        "Compact summary of learning, diversity, and worst-TF identity.",
    ),
    PlotSpec(
        "scores_projection",
        "Score projection",
        ("sequences",),
        ("plot__scores__projection.{ext}",),
        "summary",
        "Projection of min-per-TF norm vs harmonic mean for sampled sequences.",
    ),
    PlotSpec(
        "elites_nn_distance",
        "Elite nearest-neighbor distance",
        ("elites",),
        ("plot__elites__nn_distance.{ext}",),
        "summary",
        "Nearest-neighbor distance distribution for elites in TFBS-core space.",
    ),
    PlotSpec(
        "overlap_panel",
        "Overlap panel",
        ("elites",),
        ("plot__overlap__panel.{ext}",),
        "overlap",
        "Overlap heatmap and distribution (or compact summary for large TF sets).",
    ),
    PlotSpec(
        "diag_panel",
        "Diagnostics panel",
        ("trace",),
        ("plot__diag__panel.{ext}",),
        "diagnostics",
        "Trace, rank plot, swap acceptance, and move acceptance in one panel.",
    ),
)
