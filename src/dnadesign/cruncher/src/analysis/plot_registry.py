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
        "opt_trajectory",
        "Optimization trajectory",
        ("sequences",),
        ("plot__opt_trajectory.{ext}",),
        "summary",
        "Causal particle lineage view in TF score-space with random-baseline context.",
    ),
    PlotSpec(
        "opt_trajectory_sweep",
        "Optimization trajectory over sweeps",
        ("sequences",),
        ("plot__opt_trajectory_sweep.{ext}",),
        "diagnostics",
        "Cold-slot progression over sweeps with lineage handoff markers and bottleneck-TF coloring.",
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
