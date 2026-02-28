"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/analyze/rendering.py

Orchestrate analysis plot and video rendering for a resolved run context.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

from dnadesign.cruncher.app.analyze.execution import AnalysisRunExecutionContext
from dnadesign.cruncher.app.analyze.plot_resolver import AnalysisPlotFunctions
from dnadesign.cruncher.app.analyze.plotting import (
    _prepare_analysis_plot_dir,
    _render_fimo_analysis_plot,
    _render_static_analysis_plots,
    _render_trajectory_analysis_plots,
    _render_trajectory_video_plot,
)
from dnadesign.cruncher.app.analyze.run_context import AnalysisRunContext


@dataclass(frozen=True)
class AnalysisRenderResult:
    plot_entries: list[dict[str, object]]
    plot_artifacts: list[dict[str, object]]


def render_analysis_plots(
    *,
    analysis_cfg: object,
    execution: AnalysisRunExecutionContext,
    run_context: AnalysisRunContext,
    resolved_meme_tool_path: object,
    plotters: AnalysisPlotFunctions,
) -> AnalysisRenderResult:
    _prepare_analysis_plot_dir(execution.run_dir)
    plot_entries: list[dict[str, object]] = []
    plot_artifacts: list[dict[str, object]] = []

    _render_trajectory_analysis_plots(
        run_dir=execution.run_dir,
        tmp_root=execution.tmp_root,
        plot_format=run_context.plot_format,
        plot_kwargs=run_context.plot_kwargs,
        trajectory_lines_df=run_context.computed.trajectory_lines_df,
        baseline_plot_df=run_context.computed.baseline_plot_df,
        elites_plot_df=run_context.computed.elites_plot_df,
        tf_names=execution.tf_names,
        score_space_ctx=run_context.score_space_ctx,
        analysis_cfg=analysis_cfg,
        objective_from_manifest=run_context.computed.objective_from_manifest,
        optimizer_stats=execution.optimizer_stats,
        sample_meta=execution.sample_meta,
        plot_entries=plot_entries,
        plot_artifacts=plot_artifacts,
        plot_chain_trajectory_sweep=plotters.plot_chain_trajectory_sweep,
        plot_elite_score_space_context=plotters.plot_elite_score_space_context,
    )
    _render_trajectory_video_plot(
        run_dir=execution.run_dir,
        tmp_root=execution.tmp_root,
        trajectory_df=run_context.computed.trajectory_df,
        tf_names=execution.tf_names,
        pwms=execution.pwms,
        analysis_cfg=analysis_cfg,
        bidirectional=run_context.computed.bidirectional,
        pwm_pseudocounts=run_context.computed.pwm_pseudocounts,
        log_odds_clip=run_context.computed.log_odds_clip,
        plot_entries=plot_entries,
        plot_artifacts=plot_artifacts,
    )
    _render_static_analysis_plots(
        run_dir=execution.run_dir,
        tmp_root=execution.tmp_root,
        plot_format=run_context.plot_format,
        plot_kwargs=run_context.plot_kwargs,
        nn_df=run_context.computed.nn_df,
        elites_df=execution.elites_df,
        elites_plot_df=run_context.computed.elites_plot_df,
        hits_df=execution.hits_df,
        tf_names=execution.tf_names,
        pwms=execution.pwms,
        baseline_nn=run_context.computed.baseline_nn,
        objective_from_manifest=run_context.computed.objective_from_manifest,
        trace_idata=execution.trace_idata,
        optimizer_stats=execution.optimizer_stats,
        analysis_cfg=analysis_cfg,
        plot_entries=plot_entries,
        plot_artifacts=plot_artifacts,
        plot_elites_nn_distance=plotters.plot_elites_nn_distance,
        plot_elites_showcase=plotters.plot_elites_showcase,
        plot_health_panel=plotters.plot_health_panel,
    )
    _render_fimo_analysis_plot(
        run_dir=execution.run_dir,
        tmp_root=execution.tmp_root,
        plot_format=run_context.plot_format,
        plot_kwargs=run_context.plot_kwargs,
        analysis_cfg=analysis_cfg,
        trajectory_df=run_context.computed.trajectory_df,
        tf_names=execution.tf_names,
        pwms=execution.pwms,
        bidirectional=run_context.computed.bidirectional,
        resolved_meme_tool_path=resolved_meme_tool_path,
        objective_from_manifest=run_context.computed.objective_from_manifest,
        plot_entries=plot_entries,
        plot_artifacts=plot_artifacts,
        plot_optimizer_vs_fimo=plotters.plot_optimizer_vs_fimo,
    )
    return AnalysisRenderResult(plot_entries=plot_entries, plot_artifacts=plot_artifacts)


__all__ = ["AnalysisRenderResult", "render_analysis_plots"]
