"""Registry-backed aggregate plots for DenseGen axis probe reviews."""

from __future__ import annotations

from .contracts import ProbeAggregatePlotSpec, build_probe_aggregate_plot_registry
from .writer import write_probe_aggregate_plots

__all__ = [
    "ProbeAggregatePlotSpec",
    "build_probe_aggregate_plot_registry",
    "write_probe_aggregate_plots",
]
