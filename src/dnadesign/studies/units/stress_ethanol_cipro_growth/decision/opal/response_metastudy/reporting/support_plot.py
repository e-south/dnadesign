"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/reporting/support_plot.py

Candidate response-shape support plot.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..core.contracts import RecommendationThresholds
from .plot_style import save_metastudy_figure
from .plot_vocabulary import target_view_label


def write_candidate_logic_support(
    support: pd.DataFrame,
    path: Path,
    *,
    thresholds: RecommendationThresholds,
) -> None:
    data = support.copy()
    data["plot_count"] = np.log10(data["candidate_count"].astype(float) + 1.0)
    data["Target view"] = data["selection_view_id"].map(target_view_label)
    fig = plt.figure(figsize=(8.0, 4.8))
    ax = sns.lineplot(
        data=data,
        x="logic_threshold",
        y="plot_count",
        hue="Target view",
        marker="o",
    )
    ax.axvline(
        thresholds.min_target_view_median_logic,
        color="#9A3324",
        linestyle="--",
        linewidth=1.1,
        label="Review guardrail",
    )
    ax.axhline(
        np.log10(thresholds.min_effective_topk + 1.0),
        color="#555555",
        linestyle=":",
        linewidth=1.0,
        label="Top-6 capacity",
    )
    count_ticks = np.asarray([0, 1, 10, 100, 1000, 10000, 100000], dtype=float)
    ax.set_yticks(np.log10(count_ticks + 1.0), labels=[f"{int(value):,}" for value in count_ticks])
    ax.set_xlabel("Minimum SFXI logic fidelity")
    ax.set_ylabel("Predicted candidates meeting threshold")
    ax.set_title("Candidate support for each stress-response shape")
    ax.set_box_aspect(1)
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    save_metastudy_figure(fig, path)
