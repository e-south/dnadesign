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
        "elite_score_space_context",
        "Elite score-space context",
        ("sequences",),
        ("elite_score_space_context.{ext}",),
        "summary",
        "Selected elite score-space context with random baseline and theoretical consensus maxima references.",
    ),
    PlotSpec(
        "chain_trajectory_sweep",
        "Chain trajectory over sweeps",
        ("sequences",),
        ("chain_trajectory_sweep.{ext}",),
        "diagnostics",
        "Per-chain joint objective progression over sweeps with tune/cooling context when available.",
    ),
    PlotSpec(
        "chain_trajectory_video",
        "Chain trajectory video",
        ("sequences",),
        ("chain_trajectory_video.mp4",),
        "diagnostics",
        "Best-chain trajectory video with optional monotonic best-so-far timeline rendering.",
    ),
    PlotSpec(
        "elites_nn_distance",
        "Elite nearest-neighbor distance",
        ("elites",),
        ("elites_nn_distance.{ext}",),
        "summary",
        (
            "Elite diversity panel combining motif-core context with full-sequence distance diagnostics; "
            "baseline context is included when baseline hits are available."
        ),
    ),
    PlotSpec(
        "elites_showcase",
        "Elites showcase",
        ("elites",),
        ("elites_showcase.{ext}",),
        "summary",
        "Baserender-backed motif placement panels (sense/antisense + windows + logos) per elite.",
    ),
    PlotSpec(
        "health_panel",
        "Health panel",
        ("trace",),
        ("health_panel.{ext}",),
        "diagnostics",
        "MH acceptance dynamics and move-mix diagnostics over sweeps.",
    ),
    PlotSpec(
        "optimizer_vs_fimo",
        "Optimizer vs FIMO",
        ("sequences",),
        ("optimizer_vs_fimo.{ext}",),
        "diagnostics",
        "Descriptive concordance between Cruncher joint score and FIMO weakest-TF score.",
    ),
)
