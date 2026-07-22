"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/tfbs/stage_b/learning_loop_baselines/plots/canvas.py

Canvas helpers for TFBS learning-loop review plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def save_review_figure(fig: Any, path: Path) -> None:
    """Save a review figure without inheriting hostile global matplotlib save state."""

    fig.patch.set_facecolor("white")
    fig.patch.set_edgecolor("white")
    for ax in fig.axes:
        ax.set_facecolor("white")
    fig.savefig(
        path,
        dpi=160,
        facecolor="white",
        edgecolor="white",
        bbox_inches=None,
        pad_inches=0.0,
    )
