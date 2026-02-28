"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/analyze/plotting_trajectory.py

Render trajectory analysis plots and optional trajectory video output.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dnadesign.cruncher.analysis.layout import analysis_plot_path
from dnadesign.cruncher.app.analyze.plotting_registry import _record_analysis_plot
from dnadesign.cruncher.app.analyze_score_space import _ScoreSpaceContext

__all__ = ["_render_trajectory_analysis_plots", "_render_trajectory_video_plot"]


def _render_trajectory_analysis_plots(
    *,
    run_dir: Path,
    tmp_root: Path,
    plot_format: str,
    plot_kwargs: dict[str, object],
    trajectory_lines_df: pd.DataFrame,
    baseline_plot_df: pd.DataFrame,
    elites_plot_df: pd.DataFrame,
    tf_names: list[str],
    score_space_ctx: _ScoreSpaceContext,
    analysis_cfg: object,
    objective_from_manifest: dict[str, object],
    optimizer_stats: dict[str, object] | None,
    sample_meta: object,
    plot_entries: list[dict[str, object]],
    plot_artifacts: list[dict[str, object]],
    plot_chain_trajectory_sweep: object,
    plot_elite_score_space_context: object,
) -> None:
    plot_trajectory_path = analysis_plot_path(tmp_root, "elite_score_space_context", plot_format)
    plot_trajectory_sweep_path = analysis_plot_path(tmp_root, "chain_trajectory_sweep", plot_format)
    if trajectory_lines_df.empty:
        _record_analysis_plot(
            plot_entries=plot_entries,
            plot_artifacts=plot_artifacts,
            spec_key="elite_score_space_context",
            output=plot_trajectory_path,
            generated=False,
            skip_reason="trajectory table is empty",
            run_dir=run_dir,
        )
        _record_analysis_plot(
            plot_entries=plot_entries,
            plot_artifacts=plot_artifacts,
            spec_key="chain_trajectory_sweep",
            output=plot_trajectory_sweep_path,
            generated=False,
            skip_reason="trajectory table is empty",
            run_dir=run_dir,
        )
        return

    plot_elite_score_space_context(
        trajectory_df=trajectory_lines_df,
        baseline_df=baseline_plot_df,
        elites_df=elites_plot_df,
        tf_pair=score_space_ctx.trajectory_tf_pair,
        scatter_scale=score_space_ctx.trajectory_scale,
        consensus_anchors=score_space_ctx.consensus_anchors,
        objective_caption=score_space_ctx.objective_caption,
        out_path=plot_trajectory_path,
        retain_elites=analysis_cfg.trajectory_scatter_retain_elites,
        score_space_mode=score_space_ctx.mode,
        tf_names=tf_names,
        tf_pairs_grid=score_space_ctx.pairs if score_space_ctx.mode == "all_pairs_grid" else None,
        consensus_anchors_by_pair=score_space_ctx.consensus_anchors_by_pair,
        **plot_kwargs,
    )
    _record_analysis_plot(
        plot_entries=plot_entries,
        plot_artifacts=plot_artifacts,
        spec_key="elite_score_space_context",
        output=plot_trajectory_path,
        generated=True,
        skip_reason=None,
        run_dir=run_dir,
    )

    plot_chain_trajectory_sweep(
        trajectory_df=trajectory_lines_df,
        y_column=str(analysis_cfg.trajectory_sweep_y_column),
        y_mode=str(analysis_cfg.trajectory_sweep_mode),
        objective_config=objective_from_manifest,
        cooling_config=optimizer_stats.get("mcmc_cooling") if isinstance(optimizer_stats, dict) else None,
        tune_sweeps=sample_meta.tune,
        objective_caption=score_space_ctx.objective_caption,
        out_path=plot_trajectory_sweep_path,
        stride=analysis_cfg.trajectory_stride,
        alpha_min=analysis_cfg.trajectory_particle_alpha_min,
        alpha_max=analysis_cfg.trajectory_particle_alpha_max,
        chain_overlay=analysis_cfg.trajectory_chain_overlay,
        summary_overlay=analysis_cfg.trajectory_summary_overlay,
        **plot_kwargs,
    )
    _record_analysis_plot(
        plot_entries=plot_entries,
        plot_artifacts=plot_artifacts,
        spec_key="chain_trajectory_sweep",
        output=plot_trajectory_sweep_path,
        generated=True,
        skip_reason=None,
        run_dir=run_dir,
    )


def _render_trajectory_video_plot(
    *,
    run_dir: Path,
    tmp_root: Path,
    trajectory_df: pd.DataFrame,
    tf_names: list[str],
    pwms: dict[str, object],
    analysis_cfg: object,
    bidirectional: bool,
    pwm_pseudocounts: float,
    log_odds_clip: float | None,
    plot_entries: list[dict[str, object]],
    plot_artifacts: list[dict[str, object]],
) -> None:
    video_name = str(analysis_cfg.trajectory_video.output_name)
    video_stem = Path(video_name).stem
    video_ext = Path(video_name).suffix.lstrip(".")
    if not video_ext:
        video_ext = "mp4"
    plot_video_path = analysis_plot_path(tmp_root, video_stem, video_ext)
    if not analysis_cfg.trajectory_video.enabled:
        _record_analysis_plot(
            plot_entries=plot_entries,
            plot_artifacts=plot_artifacts,
            spec_key="chain_trajectory_video",
            output=plot_video_path,
            generated=False,
            skip_reason="analysis.trajectory_video.enabled=false",
            run_dir=run_dir,
        )
        return
    if trajectory_df.empty:
        _record_analysis_plot(
            plot_entries=plot_entries,
            plot_artifacts=plot_artifacts,
            spec_key="chain_trajectory_video",
            output=plot_video_path,
            generated=False,
            skip_reason="trajectory table is empty",
            run_dir=run_dir,
        )
        return

    from dnadesign.cruncher.analysis.trajectory_video import render_chain_trajectory_video

    render_chain_trajectory_video(
        trajectory_df=trajectory_df,
        tf_names=tf_names,
        pwms=pwms,
        out_path=plot_video_path,
        config=analysis_cfg.trajectory_video,
        bidirectional=bidirectional,
        pwm_pseudocounts=float(pwm_pseudocounts),
        log_odds_clip=log_odds_clip,
        tmp_root=tmp_root / "_trajectory_video_tmp",
    )
    _record_analysis_plot(
        plot_entries=plot_entries,
        plot_artifacts=plot_artifacts,
        spec_key="chain_trajectory_video",
        output=plot_video_path,
        generated=True,
        skip_reason=None,
        run_dir=run_dir,
    )
