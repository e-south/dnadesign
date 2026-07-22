"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/reporting/rmf_contract_plot.py

Plot the state-cardinality pressure test for generic RMF use.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .plot_helpers import require_columns
from .plot_style import save_metastudy_figure


def write_rmf_cardinality_pressure(frame: pd.DataFrame, path: Path) -> None:
    required = {
        "state_count",
        "mask_topology",
        "response_separation_bias",
        "on_magnitude_floor_bias",
        "off_magnitude_ceiling_bias",
    }
    require_columns(frame, required, context="RMF cardinality pressure plot")
    components = (
        ("response_separation_bias", "Response-ordering bias"),
        ("on_magnitude_floor_bias", "ON-floor bias"),
        ("off_magnitude_ceiling_bias", "OFF-ceiling bias"),
    )
    topology_order = ("one ON", "balanced", "one OFF")
    colors = {"one ON": "#2f5597", "balanced": "#548235", "one OFF": "#c55a11"}
    markers = {"one ON": "o", "balanced": "s", "one OFF": "^"}
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.7), sharex=True, constrained_layout=True)
    for axis, (column, label) in zip(axes, components, strict=True):
        for topology in topology_order:
            rows = frame.loc[frame["mask_topology"].eq(topology)].sort_values("state_count")
            if rows.empty:
                raise ValueError(f"RMF cardinality pressure lacks target topology {topology!r}.")
            axis.plot(
                rows["state_count"],
                rows[column],
                color=colors[topology],
                marker=markers[topology],
                linewidth=1.8,
                label=topology,
                zorder=3,
            )
        axis.axhline(0.0, color="#6b7280", linestyle="--", linewidth=0.9, zorder=1)
        axis.set_title(label)
        axis.set_xlabel("Number of assay states")
        axis.set_xticks(sorted(frame["state_count"].unique()))
        axis.set_box_aspect(1.0)
    axes[0].set_ylabel("Mean bias from independent noise (log2 units)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=3,
        title="Target-mask composition",
    )
    save_metastudy_figure(fig, path)


__all__ = ["write_rmf_cardinality_pressure"]
