"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/analyze_workflow.py

Analyze sampling runs and produce summary reports.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from dnadesign.cruncher.app.analyze.archive import _prune_latest_analysis_artifacts
from dnadesign.cruncher.app.analyze.execution import (
    AnalysisRunExecutionContext,
    resolve_analysis_run_execution_context,
)
from dnadesign.cruncher.app.analyze.metadata import _get_version
from dnadesign.cruncher.app.analyze.plan import resolve_analysis_plan
from dnadesign.cruncher.app.analyze.plot_resolver import AnalysisPlotFunctions, resolve_analysis_plot_functions
from dnadesign.cruncher.app.analyze.publish import publish_analysis_outputs
from dnadesign.cruncher.app.analyze.rendering import render_analysis_plots
from dnadesign.cruncher.app.analyze.run_context import AnalysisRunContext, resolve_analysis_run_context
from dnadesign.cruncher.app.analyze.run_resolution import _resolve_run_names
from dnadesign.cruncher.app.analyze.staging import (
    analysis_managed_paths,
    delete_path,
    finalize_analysis_root,
    prepare_analysis_root,
)
from dnadesign.cruncher.app.analyze_support import (
    _load_elites_meta,
    _write_analysis_used,
)
from dnadesign.cruncher.artifacts.entries import append_artifacts
from dnadesign.cruncher.artifacts.manifest import write_manifest
from dnadesign.cruncher.config.schema_v3 import CruncherConfig
from dnadesign.cruncher.integrations.meme_suite import resolve_tool_path
from dnadesign.cruncher.utils.arviz_cache import ensure_arviz_data_dir
from dnadesign.cruncher.utils.paths import resolve_catalog_root
from dnadesign.cruncher.viz.mpl import ensure_mpl_cache

logger = logging.getLogger(__name__)
_ANALYZE_SUPPORT_EXPORTS = (_load_elites_meta,)

__all__ = ["run_analyze"]


def _finalize_analyze_root_with_recovery(
    *,
    analysis_root_path: Path,
    analysis_id: str,
    tmp_root: Path,
    archive: bool,
) -> None:
    prev_root = None
    try:
        prev_root = prepare_analysis_root(
            analysis_root_path,
            analysis_id=analysis_id,
        )
        finalize_analysis_root(
            analysis_root_path,
            tmp_root,
            archive=archive,
            prev_root=prev_root,
        )
    except Exception:
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)
        if prev_root is not None and prev_root.exists():
            for path in analysis_managed_paths(analysis_root_path):
                delete_path(path)
            for child in prev_root.iterdir():
                shutil.move(str(child), analysis_root_path / child.name)
            shutil.rmtree(prev_root, ignore_errors=True)
        raise


def _run_analyze_for_run(
    *,
    analysis_cfg: object,
    execution: AnalysisRunExecutionContext,
    resolved_meme_tool_path: object,
    plotters: AnalysisPlotFunctions,
) -> Path:
    run_context: AnalysisRunContext = resolve_analysis_run_context(
        analysis_cfg=analysis_cfg,
        execution=execution,
    )

    _write_analysis_used(
        execution.analysis_used_file,
        run_context.analysis_cfg_payload,
        execution.analysis_id,
        execution.run_name,
        extras={
            "tf_pair_mode": analysis_cfg.pairwise,
            "tf_pair_selected": (
                list(run_context.score_space_ctx.focus_pair) if run_context.score_space_ctx.focus_pair else None
            ),
            "trajectory_tf_pair": list(run_context.score_space_ctx.trajectory_tf_pair),
            "score_space_mode": run_context.score_space_ctx.mode,
            "score_space_pairs": [list(pair) for pair in run_context.score_space_ctx.pairs],
            "trajectory_scatter_scale": run_context.score_space_ctx.trajectory_scale,
            "trajectory_scatter_retain_elites": analysis_cfg.trajectory_scatter_retain_elites,
            "trajectory_sweep_y_column": analysis_cfg.trajectory_sweep_y_column,
            "trajectory_sweep_mode": analysis_cfg.trajectory_sweep_mode,
            "trajectory_summary_overlay": analysis_cfg.trajectory_summary_overlay,
            "trajectory_video_enabled": analysis_cfg.trajectory_video.enabled,
            "trajectory_video_timeline_mode": analysis_cfg.trajectory_video.timeline_mode,
            "trajectory_video_duration_sec": analysis_cfg.trajectory_video.playback.target_duration_sec,
            "trajectory_video_fps": analysis_cfg.trajectory_video.playback.fps,
            "require_random_baseline": execution.require_random_baseline,
        },
    )

    rendered = render_analysis_plots(
        analysis_cfg=analysis_cfg,
        execution=execution,
        run_context=run_context,
        resolved_meme_tool_path=resolved_meme_tool_path,
        plotters=plotters,
    )

    analysis_artifacts = publish_analysis_outputs(
        analysis_id=execution.analysis_id,
        created_at=execution.created_at,
        run_name=execution.run_name,
        analysis_root_path=execution.analysis_root_path,
        tmp_root=execution.tmp_root,
        run_dir=execution.run_dir,
        analysis_used_file=execution.analysis_used_file,
        analysis_cfg_payload=run_context.analysis_cfg_payload,
        tf_names=execution.tf_names,
        diagnostics_payload=run_context.computed.diagnostics_payload,
        objective_components=run_context.computed.objective_components,
        overlap_summary=run_context.computed.overlap_summary,
        table_paths=run_context.computed.table_paths,
        mmr_sweep_enabled=analysis_cfg.mmr_sweep.enabled,
        plot_entries=rendered.plot_entries,
        plot_artifacts=rendered.plot_artifacts,
        version=_get_version(),
    )

    _finalize_analyze_root_with_recovery(
        analysis_root_path=execution.analysis_root_path,
        analysis_id=execution.analysis_id,
        tmp_root=execution.tmp_root,
        archive=analysis_cfg.archive,
    )

    _prune_latest_analysis_artifacts(execution.manifest)
    append_artifacts(execution.manifest, analysis_artifacts)
    write_manifest(execution.run_dir, execution.manifest)

    return execution.analysis_root_path


def run_analyze(
    cfg: CruncherConfig,
    config_path: Path,
    *,
    runs_override: list[str] | None = None,
    use_latest: bool = False,
) -> list[Path]:
    plan = resolve_analysis_plan(cfg)
    analysis_cfg = plan.analysis_cfg
    if not analysis_cfg.enabled:
        raise ValueError("analysis.enabled=false; set analysis.enabled=true to run analysis.")
    runs = _resolve_run_names(
        cfg,
        config_path,
        analysis_cfg=analysis_cfg,
        runs_override=runs_override,
        use_latest=use_latest,
    )
    catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)
    ensure_mpl_cache(catalog_root)
    ensure_arviz_data_dir(catalog_root)
    plotters = resolve_analysis_plot_functions()
    resolved_meme_tool_path = resolve_tool_path(cfg.discover.tool_path, config_path=config_path)

    results: list[Path] = []
    for run_name in runs:
        execution = resolve_analysis_run_execution_context(
            cfg=cfg,
            config_path=config_path,
            run_name=run_name,
        )
        results.append(
            _run_analyze_for_run(
                analysis_cfg=analysis_cfg,
                execution=execution,
                resolved_meme_tool_path=resolved_meme_tool_path,
                plotters=plotters,
            )
        )
    return results
