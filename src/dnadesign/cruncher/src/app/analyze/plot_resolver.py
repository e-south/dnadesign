"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/analyze/plot_resolver.py

Resolve analysis plot callables lazily after plotting cache initialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisPlotFunctions:
    plot_elites_nn_distance: object
    plot_elites_showcase: object
    plot_optimizer_vs_fimo: object
    plot_health_panel: object
    plot_chain_trajectory_sweep: object
    plot_elite_score_space_context: object


def resolve_analysis_plot_functions() -> AnalysisPlotFunctions:
    from dnadesign.cruncher.analysis.plots.elites_nn_distance import plot_elites_nn_distance
    from dnadesign.cruncher.analysis.plots.elites_showcase import plot_elites_showcase
    from dnadesign.cruncher.analysis.plots.fimo_concordance import plot_optimizer_vs_fimo
    from dnadesign.cruncher.analysis.plots.health_panel import plot_health_panel
    from dnadesign.cruncher.analysis.plots.trajectory_score_space_plot import plot_elite_score_space_context
    from dnadesign.cruncher.analysis.plots.trajectory_sweep import plot_chain_trajectory_sweep

    return AnalysisPlotFunctions(
        plot_elites_nn_distance=plot_elites_nn_distance,
        plot_elites_showcase=plot_elites_showcase,
        plot_optimizer_vs_fimo=plot_optimizer_vs_fimo,
        plot_health_panel=plot_health_panel,
        plot_chain_trajectory_sweep=plot_chain_trajectory_sweep,
        plot_elite_score_space_context=plot_elite_score_space_context,
    )


__all__ = ["AnalysisPlotFunctions", "resolve_analysis_plot_functions"]
