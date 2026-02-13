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
        "chain_trajectory_scatter",
        "Chain trajectory scatter",
        ("sequences",),
        ("plot__chain_trajectory_scatter.{ext}",),
        "summary",
        "Best-so-far chain lineage updates in TF score-space with baseline and selected elite overlays.",
    ),
    PlotSpec(
        "chain_trajectory_sweep",
        "Chain trajectory over sweeps",
        ("sequences",),
        ("plot__chain_trajectory_sweep.{ext}",),
        "diagnostics",
        "Per-chain joint objective progression over sweeps with tune/cooling context when available.",
    ),
    PlotSpec(
        "elites_nn_distance",
        "Elite nearest-neighbor distance",
        ("elites", "baseline_hits"),
        ("plot__elites_nn_distance.{ext}",),
        "summary",
        "Elite diversity panel combining motif-core context with full-sequence distance diagnostics.",
    ),
    PlotSpec(
        "elites_showcase",
        "Elites showcase",
        ("elites",),
        ("plot__elites_showcase.{ext}",),
        "summary",
        "Baserender-backed motif placement panels (sense/antisense + windows + logos) per elite.",
    ),
    PlotSpec(
        "health_panel",
        "Health panel",
        ("trace",),
        ("plot__health_panel.{ext}",),
        "diagnostics",
        "MH acceptance dynamics and move-mix diagnostics over sweeps.",
    ),
    PlotSpec(
        "optimizer_vs_fimo",
        "Optimizer vs FIMO",
        ("sequences",),
        ("plot__optimizer_vs_fimo.{ext}",),
        "diagnostics",
        "Descriptive concordance between Cruncher joint score and FIMO weakest-TF score.",
    ),
)
