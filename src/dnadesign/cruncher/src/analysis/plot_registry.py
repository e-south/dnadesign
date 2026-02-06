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
        "run_summary",
        "Run summary",
        ("sequences", "elites", "baseline"),
        ("plots/run_summary.{ext}",),
        "summary",
        "Single-page summary of learning, outcome, and diversity.",
    ),
    PlotSpec(
        "opt_trajectory",
        "Optimization trajectory",
        ("sequences", "baseline"),
        ("plots/opt_trajectory.{ext}",),
        "summary",
        "Trajectory in score space with a baseline cloud for context.",
    ),
    PlotSpec(
        "elites_nn_distance",
        "Elite nearest-neighbor distance",
        ("elites", "baseline_hits"),
        ("plots/elites_nn_distance.{ext}",),
        "summary",
        "Nearest-neighbor distance distribution for elites in TFBS-core space.",
    ),
    PlotSpec(
        "overlap_panel",
        "Overlap panel",
        ("elites",),
        ("plots/overlap_panel.{ext}",),
        "overlap",
        "Overlap heatmap and distribution (or compact summary for large TF sets).",
    ),
    PlotSpec(
        "health_panel",
        "Health panel",
        ("trace",),
        ("plots/health_panel.{ext}",),
        "diagnostics",
        "Swap acceptance and move acceptance summary.",
    ),
)
