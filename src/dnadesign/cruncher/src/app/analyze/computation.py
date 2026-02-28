"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/analyze/computation.py

Compute and persist analysis tables/metrics used by reports and plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from dnadesign.cruncher.analysis.diagnostics import summarize_sampling_diagnostics
from dnadesign.cruncher.analysis.diversity import compute_baseline_nn_distances
from dnadesign.cruncher.analysis.layout import analysis_table_path
from dnadesign.cruncher.analysis.mmr_sweep_service import run_mmr_sweep_for_run
from dnadesign.cruncher.analysis.objective import compute_objective_components
from dnadesign.cruncher.analysis.overlap import compute_overlap_tables
from dnadesign.cruncher.analysis.trajectory import build_chain_trajectory_points
from dnadesign.cruncher.app.analyze.metadata import SampleMeta
from dnadesign.cruncher.app.analyze_score_space import (
    _project_trajectory_views_with_cleanup,
    _resolve_objective_projection_inputs,
)
from dnadesign.cruncher.app.analyze_support import (
    _resolve_baseline_seed,
    _summarize_elites_mmr,
    _write_json,
    _write_table,
)


@dataclass(frozen=True)
class AnalysisTablesAndMetrics:
    objective_from_manifest: dict[str, object]
    bidirectional: bool
    pwm_pseudocounts: object
    log_odds_clip: object
    retain_sequences: bool
    trajectory_df: pd.DataFrame
    trajectory_lines_df: pd.DataFrame
    baseline_plot_df: pd.DataFrame
    elites_plot_df: pd.DataFrame
    nn_df: pd.DataFrame
    baseline_nn: pd.DataFrame
    diagnostics_payload: dict[str, object]
    objective_components: dict[str, object]
    overlap_summary: dict[str, object]
    table_paths: dict[str, Path]


def compute_analysis_tables_and_metrics(
    *,
    tmp_root: Path,
    run_dir: Path,
    analysis_cfg: object,
    sequences_df: pd.DataFrame,
    elites_df: pd.DataFrame,
    hits_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    baseline_hits_df: pd.DataFrame,
    trace_idata: object | None,
    tf_names: list[str],
    pwms: dict[str, Any],
    manifest: dict[str, object],
    sample_meta: SampleMeta,
    used_cfg: dict[str, object],
    optimizer_stats: dict[str, object] | None,
    elites_meta: dict[str, object],
) -> AnalysisTablesAndMetrics:
    table_ext = analysis_cfg.table_format
    score_summary_path = analysis_table_path(tmp_root, "scores_summary", table_ext)
    topk_path = analysis_table_path(tmp_root, "elites_topk", table_ext)
    joint_metrics_path = analysis_table_path(tmp_root, "metrics_joint", table_ext)
    overlap_pair_path = analysis_table_path(tmp_root, "overlap_pair_summary", table_ext)
    overlap_elite_path = analysis_table_path(tmp_root, "overlap_per_elite", table_ext)
    diagnostics_path = analysis_table_path(tmp_root, "diagnostics_summary", "json")
    objective_path = analysis_table_path(tmp_root, "objective_components", "json")
    elites_mmr_path = analysis_table_path(tmp_root, "elites_mmr_summary", table_ext)
    elites_mmr_sweep_path = analysis_table_path(tmp_root, "elites_mmr_sweep", table_ext)
    nn_distance_path = analysis_table_path(tmp_root, "elites_nn_distance", table_ext)
    trajectory_path = analysis_table_path(tmp_root, "chain_trajectory_points", table_ext)
    trajectory_lines_path = analysis_table_path(tmp_root, "chain_trajectory_lines", table_ext)

    from dnadesign.cruncher.analysis.plots.summary import (
        score_frame_from_df,
        write_elite_topk,
        write_joint_metrics,
        write_score_summary,
    )

    score_df = score_frame_from_df(sequences_df, tf_names)
    write_score_summary(score_df=score_df, tf_names=tf_names, out_path=score_summary_path)

    top_k = sample_meta.top_k if sample_meta.top_k else len(elites_df)
    write_elite_topk(elites_df=elites_df, tf_names=tf_names, out_path=topk_path, top_k=top_k)
    write_joint_metrics(elites_df=elites_df, tf_names=tf_names, out_path=joint_metrics_path)

    objective_from_manifest, bidirectional, pwm_pseudocounts, log_odds_clip, retain_sequences, beta_ladder = (
        _resolve_objective_projection_inputs(
            manifest=manifest,
            sample_meta=sample_meta,
            used_cfg=used_cfg,
            elites_df=elites_df,
            optimizer_stats=optimizer_stats,
        )
    )

    trajectory_df, baseline_plot_df, elites_plot_df = _project_trajectory_views_with_cleanup(
        tmp_root=tmp_root,
        sequences_df=sequences_df,
        baseline_df=baseline_df,
        elites_df=elites_df,
        tf_names=tf_names,
        pwms=pwms,
        analysis_cfg=analysis_cfg,
        objective_from_manifest=objective_from_manifest,
        bidirectional=bidirectional,
        pseudocounts_raw=pwm_pseudocounts,
        log_odds_clip_raw=log_odds_clip,
        beta_ladder=beta_ladder,
        retain_sequences=retain_sequences,
    )
    _write_table(trajectory_df, trajectory_path)
    trajectory_lines_df = build_chain_trajectory_points(
        trajectory_df,
        max_points=analysis_cfg.max_points,
        retain_sequences=retain_sequences,
    )
    _write_table(trajectory_lines_df, trajectory_lines_path)

    overlap_summary_df, elite_overlap_df, overlap_summary = compute_overlap_tables(
        elites_df, hits_df, tf_names, include_sequences=False
    )
    _write_table(overlap_summary_df, overlap_pair_path)
    _write_table(elite_overlap_df, overlap_elite_path)

    diagnostics_payload = summarize_sampling_diagnostics(
        trace_idata=trace_idata,
        sequences_df=sequences_df,
        elites_df=elites_df,
        elites_hits_df=hits_df,
        tf_names=tf_names,
        optimizer=manifest.get("optimizer"),
        optimizer_stats=optimizer_stats,
        mode=sample_meta.mode,
        optimizer_kind=sample_meta.optimizer_kind,
        sample_meta={
            "chains": sample_meta.chains,
            "draws": sample_meta.draws,
            "tune": sample_meta.tune,
            "mode": sample_meta.mode,
            "optimizer_kind": sample_meta.optimizer_kind,
            "top_k": sample_meta.top_k,
            "dsdna_canonicalize": sample_meta.bidirectional,
        },
        trace_required=False,
        overlap_summary=overlap_summary,
    )
    _write_json(diagnostics_path, diagnostics_payload)

    objective_components = compute_objective_components(
        sequences_df=sequences_df,
        tf_names=tf_names,
        top_k=sample_meta.top_k,
        overlap_total_bp_median=overlap_summary.get("overlap_total_bp_median"),
    )
    _write_json(objective_path, objective_components)

    elites_mmr_df, nn_df = _summarize_elites_mmr(
        elites_df,
        hits_df,
        sequences_df,
        elites_meta,
        tf_names,
        pwms,
        bidirectional=bool(sample_meta.bidirectional),
    )
    _write_table(elites_mmr_df, elites_mmr_path)
    _write_table(nn_df, nn_distance_path)
    if analysis_cfg.mmr_sweep.enabled:
        run_mmr_sweep_for_run(
            run_dir,
            pool_size_values=analysis_cfg.mmr_sweep.pool_size_values,
            diversity_values=analysis_cfg.mmr_sweep.diversity_values,
            out_path=elites_mmr_sweep_path,
        )

    baseline_seed = _resolve_baseline_seed(baseline_df)
    baseline_nn = compute_baseline_nn_distances(
        baseline_hits_df,
        tf_names,
        pwms,
        seed=baseline_seed,
    )

    table_paths = {
        "scores_summary": score_summary_path,
        "elites_topk": topk_path,
        "metrics_joint": joint_metrics_path,
        "chain_trajectory_points": trajectory_path,
        "chain_trajectory_lines": trajectory_lines_path,
        "overlap_pair_summary": overlap_pair_path,
        "overlap_per_elite": overlap_elite_path,
        "diagnostics_summary": diagnostics_path,
        "objective_components": objective_path,
        "elites_mmr_summary": elites_mmr_path,
        "elites_mmr_sweep": elites_mmr_sweep_path,
        "elites_nn_distance": nn_distance_path,
    }
    return AnalysisTablesAndMetrics(
        objective_from_manifest=objective_from_manifest,
        bidirectional=bidirectional,
        pwm_pseudocounts=pwm_pseudocounts,
        log_odds_clip=log_odds_clip,
        retain_sequences=retain_sequences,
        trajectory_df=trajectory_df,
        trajectory_lines_df=trajectory_lines_df,
        baseline_plot_df=baseline_plot_df,
        elites_plot_df=elites_plot_df,
        nn_df=nn_df,
        baseline_nn=baseline_nn,
        diagnostics_payload=diagnostics_payload,
        objective_components=objective_components,
        overlap_summary=overlap_summary,
        table_paths=table_paths,
    )
