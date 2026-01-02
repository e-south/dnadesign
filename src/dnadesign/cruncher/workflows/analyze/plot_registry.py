"""Plot registry metadata for analysis planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PlotSpec:
    key: str
    label: str
    requires: Tuple[str, ...]


PLOT_SPECS: tuple[PlotSpec, ...] = (
    PlotSpec("trace", "Trace (per-chain)", ("trace",)),
    PlotSpec("autocorr", "Autocorrelation", ("trace",)),
    PlotSpec("convergence", "Convergence diagnostics", ("trace",)),
    PlotSpec("pair_pwm", "Pairwise PWM scores", ("tf_pair",)),
    PlotSpec("parallel_pwm", "Pairwise PWM parallel plot", ("tf_pair",)),
    PlotSpec("scatter_pwm", "Per-PWM scatter", ("tf_pair",)),
    PlotSpec("score_hist", "Per-TF score histogram", ()),
    PlotSpec("score_box", "Per-TF score boxplot", ()),
    PlotSpec("correlation_heatmap", "Score correlation heatmap", ()),
    PlotSpec("parallel_coords", "Parallel coordinates (top-K)", ("elites",)),
)
