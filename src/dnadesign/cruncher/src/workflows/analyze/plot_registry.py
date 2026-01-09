"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/workflows/analyze/plot_registry.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

from dnadesign.cruncher.config.schema_v2 import AnalysisPlotConfig


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
        "trace",
        "Trace (per-chain)",
        ("trace",),
        ("plots/diag__trace_score.png",),
        "diagnostics",
        "Score trace with posterior density for each chain.",
    ),
    PlotSpec(
        "autocorr",
        "Autocorrelation",
        ("trace",),
        ("plots/diag__autocorr_score.png",),
        "diagnostics",
        "Autocorrelation of the score trace across lags.",
    ),
    PlotSpec(
        "convergence",
        "Convergence diagnostics",
        ("trace",),
        (
            "plots/diag__convergence.txt",
            "plots/diag__rank_plot_score.png",
            "plots/diag__ess_evolution_score.png",
        ),
        "diagnostics",
        "R-hat/ESS summary and diagnostics for score convergence.",
    ),
    PlotSpec(
        "pair_pwm",
        "Pairwise PWM scores",
        ("tf_pair",),
        ("plots/pwm__pair_scores.png",),
        "pairwise",
        "Joint score distribution for a selected TF pair.",
    ),
    PlotSpec(
        "parallel_pwm",
        "Pairwise PWM parallel plot",
        ("tf_pair",),
        ("plots/pwm__parallel_scores.png",),
        "pairwise",
        "Parallel coordinates for a selected TF pair.",
    ),
    PlotSpec(
        "scatter_pwm",
        "Per-PWM scatter",
        ("tf_pair",),
        ("plots/pwm__scatter.png", "plots/pwm__scatter.pdf"),
        "pairwise",
        "Scatter of per-PWM scores for a selected TF pair.",
    ),
    PlotSpec(
        "score_hist",
        "Per-TF score histogram",
        (),
        ("plots/score__hist.png",),
        "summary",
        "Per-TF score distributions across sampled sequences.",
    ),
    PlotSpec(
        "score_box",
        "Per-TF score boxplot",
        (),
        ("plots/score__box.png",),
        "summary",
        "Per-TF score quartiles and outliers.",
    ),
    PlotSpec(
        "correlation_heatmap",
        "Score correlation heatmap",
        (),
        ("plots/score__correlation.png",),
        "summary",
        "Correlation matrix across TF score dimensions.",
    ),
    PlotSpec(
        "pairgrid",
        "Pairwise score pairgrid",
        (),
        ("plots/score__pairgrid.png",),
        "summary",
        "Pairwise projection grid across TF score dimensions.",
    ),
    PlotSpec(
        "parallel_coords",
        "Parallel coordinates (top-K)",
        ("elites",),
        ("plots/elites__parallel_coords.png",),
        "summary",
        "Parallel coordinates for top-K elite sequences.",
    ),
)


def plot_registry_rows(
    *,
    enabled: AnalysisPlotConfig | None,
    pair_available: bool | None,
    overrides: Iterable[str] | None = None,
) -> list[dict[str, str]]:
    override_set = {item for item in (overrides or [])}
    rows: list[dict[str, str]] = []
    for spec in PLOT_SPECS:
        if enabled is None:
            enabled_label = "-"
        else:
            enabled_flag = getattr(enabled, spec.key, False)
            enabled_label = "yes" if enabled_flag else "no"
            if overrides:
                if "all" in override_set:
                    enabled_label = "yes"
                elif spec.key in override_set:
                    enabled_label = "yes"
                else:
                    enabled_label = "no"
        requires = []
        if "trace" in spec.requires:
            requires.append("trace.nc")
        if "tf_pair" in spec.requires:
            requires.append("tf_pair")
        if "elites" in spec.requires:
            requires.append("elites.parquet")
        requires_label = ", ".join(requires) if requires else "-"
        if enabled is not None and "tf_pair" in spec.requires and pair_available is False and enabled_label == "yes":
            enabled_label = "missing tf_pair"
        rows.append(
            {
                "key": spec.key,
                "label": spec.label,
                "enabled": enabled_label,
                "requires": requires_label,
                "outputs": ", ".join(spec.outputs),
            }
        )
    return rows


def plot_keys() -> set[str]:
    return {spec.key for spec in PLOT_SPECS}
