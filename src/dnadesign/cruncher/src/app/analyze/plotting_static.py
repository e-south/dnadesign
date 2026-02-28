"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/analyze/plotting_static.py

Render static analysis plots and optional optimizer-vs-FIMO concordance plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from dnadesign.cruncher.analysis.fimo_concordance import build_fimo_concordance_table
from dnadesign.cruncher.analysis.layout import analysis_plot_path
from dnadesign.cruncher.app.analyze.plotting_registry import _record_analysis_plot
from dnadesign.cruncher.app.analyze_support import _objective_axis_label

__all__ = ["_render_fimo_analysis_plot", "_render_static_analysis_plots"]


def _render_static_analysis_plots(
    *,
    run_dir: Path,
    tmp_root: Path,
    plot_format: str,
    plot_kwargs: dict[str, object],
    nn_df: pd.DataFrame,
    elites_df: pd.DataFrame,
    elites_plot_df: pd.DataFrame,
    hits_df: pd.DataFrame,
    tf_names: list[str],
    pwms: dict[str, object],
    baseline_nn: Sequence[float],
    objective_from_manifest: dict[str, object],
    trace_idata: object | None,
    optimizer_stats: dict[str, object] | None,
    analysis_cfg: object,
    plot_entries: list[dict[str, object]],
    plot_artifacts: list[dict[str, object]],
    plot_elites_nn_distance: object,
    plot_elites_showcase: object,
    plot_health_panel: object,
) -> None:
    plot_nn_path = analysis_plot_path(tmp_root, "elites_nn_distance", plot_format)
    plot_elites_nn_distance(
        nn_df,
        plot_nn_path,
        elites_df=elites_plot_df,
        baseline_nn=pd.Series(baseline_nn),
        objective_config=objective_from_manifest,
        **plot_kwargs,
    )
    _record_analysis_plot(
        plot_entries=plot_entries,
        plot_artifacts=plot_artifacts,
        spec_key="elites_nn_distance",
        output=plot_nn_path,
        generated=True,
        skip_reason=None,
        run_dir=run_dir,
    )

    plot_showcase_path = analysis_plot_path(tmp_root, "elites_showcase", plot_format)
    plot_elites_showcase(
        elites_df=elites_df,
        hits_df=hits_df,
        tf_names=tf_names,
        pwms=pwms,
        out_path=plot_showcase_path,
        max_panels=int(analysis_cfg.elites_showcase.max_panels),
        **plot_kwargs,
    )
    _record_analysis_plot(
        plot_entries=plot_entries,
        plot_artifacts=plot_artifacts,
        spec_key="elites_showcase",
        output=plot_showcase_path,
        generated=True,
        skip_reason=None,
        run_dir=run_dir,
    )

    plot_health_path = analysis_plot_path(tmp_root, "health_panel", plot_format)
    if trace_idata is None:
        _record_analysis_plot(
            plot_entries=plot_entries,
            plot_artifacts=plot_artifacts,
            spec_key="health_panel",
            output=plot_health_path,
            generated=False,
            skip_reason="trace disabled",
            run_dir=run_dir,
        )
        return
    plot_health_panel(optimizer_stats, plot_health_path, **plot_kwargs)
    _record_analysis_plot(
        plot_entries=plot_entries,
        plot_artifacts=plot_artifacts,
        spec_key="health_panel",
        output=plot_health_path,
        generated=True,
        skip_reason=None,
        run_dir=run_dir,
    )


def _render_fimo_analysis_plot(
    *,
    run_dir: Path,
    tmp_root: Path,
    plot_format: str,
    plot_kwargs: dict[str, object],
    analysis_cfg: object,
    trajectory_df: pd.DataFrame,
    tf_names: list[str],
    pwms: dict[str, object],
    bidirectional: bool,
    resolved_meme_tool_path: object,
    objective_from_manifest: dict[str, object],
    plot_entries: list[dict[str, object]],
    plot_artifacts: list[dict[str, object]],
    plot_optimizer_vs_fimo: object,
) -> None:
    plot_fimo_path = analysis_plot_path(tmp_root, "optimizer_vs_fimo", plot_format)
    if not analysis_cfg.fimo_compare.enabled:
        _record_analysis_plot(
            plot_entries=plot_entries,
            plot_artifacts=plot_artifacts,
            spec_key="optimizer_vs_fimo",
            output=plot_fimo_path,
            generated=False,
            skip_reason="analysis.fimo_compare.enabled=false",
            run_dir=run_dir,
        )
        return
    fimo_tmp_dir = tmp_root / "_fimo_compare_tmp"
    try:
        concordance_df, _ = build_fimo_concordance_table(
            points_df=trajectory_df,
            tf_names=tf_names,
            pwms=pwms,
            bidirectional=bidirectional,
            threshold=1.0,
            work_dir=fimo_tmp_dir,
            tool_path=resolved_meme_tool_path,
        )
        plot_optimizer_vs_fimo(
            concordance_df,
            plot_fimo_path,
            x_label=_objective_axis_label(objective_from_manifest),
            y_label="FIMO weakest-TF score (-log10 sequence p-value)",
            title="Cruncher optimizer vs FIMO weakest-TF score",
            **plot_kwargs,
        )
        _record_analysis_plot(
            plot_entries=plot_entries,
            plot_artifacts=plot_artifacts,
            spec_key="optimizer_vs_fimo",
            output=plot_fimo_path,
            generated=True,
            skip_reason=None,
            run_dir=run_dir,
        )
    finally:
        shutil.rmtree(fimo_tmp_dir, ignore_errors=True)
