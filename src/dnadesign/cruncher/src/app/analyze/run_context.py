"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/analyze/run_context.py

Resolves run-scoped computed analysis context used by plotting and publication.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

from dnadesign.cruncher.app.analyze.computation import AnalysisTablesAndMetrics, compute_analysis_tables_and_metrics
from dnadesign.cruncher.app.analyze.execution import AnalysisRunExecutionContext
from dnadesign.cruncher.app.analyze_score_space import _resolve_score_space_context, _ScoreSpaceContext


@dataclass(frozen=True)
class AnalysisRunContext:
    execution: AnalysisRunExecutionContext
    computed: AnalysisTablesAndMetrics
    score_space_ctx: _ScoreSpaceContext
    plot_format: str
    plot_kwargs: dict[str, object]
    analysis_cfg_payload: dict[str, object]


def resolve_analysis_run_context(
    *,
    analysis_cfg: object,
    execution: AnalysisRunExecutionContext,
) -> AnalysisRunContext:
    if not execution.tf_names:
        raise ValueError("Analysis run context requires at least one TF name.")

    computed = compute_analysis_tables_and_metrics(
        tmp_root=execution.tmp_root,
        run_dir=execution.run_dir,
        analysis_cfg=analysis_cfg,
        sequences_df=execution.sequences_df,
        elites_df=execution.elites_df,
        hits_df=execution.hits_df,
        baseline_df=execution.baseline_df,
        baseline_hits_df=execution.baseline_hits_df,
        trace_idata=execution.trace_idata,
        tf_names=execution.tf_names,
        pwms=execution.pwms,
        manifest=execution.manifest,
        sample_meta=execution.sample_meta,
        used_cfg=execution.used_cfg,
        optimizer_stats=execution.optimizer_stats,
        elites_meta=execution.elites_meta,
    )
    score_space_ctx = _resolve_score_space_context(
        tf_names=execution.tf_names,
        analysis_cfg=analysis_cfg,
        elites_plot_df=computed.elites_plot_df,
        pwms=execution.pwms,
        sequences_df=execution.sequences_df,
        manifest=execution.manifest,
        objective_from_manifest=computed.objective_from_manifest,
    )
    plot_format = str(analysis_cfg.plot_format)
    plot_kwargs = {"dpi": int(analysis_cfg.plot_dpi), "png_compress_level": 9}
    analysis_cfg_payload = analysis_cfg.model_dump(mode="json")
    return AnalysisRunContext(
        execution=execution,
        computed=computed,
        score_space_ctx=score_space_ctx,
        plot_format=plot_format,
        plot_kwargs=plot_kwargs,
        analysis_cfg_payload=analysis_cfg_payload,
    )


__all__ = ["AnalysisRunContext", "resolve_analysis_run_context"]
