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


@dataclass(frozen=True)
class _AnalysisTablePaths:
    scores_summary: Path
    elites_topk: Path
    metrics_joint: Path
    overlap_pair_summary: Path
    overlap_per_elite: Path
    diagnostics_summary: Path
    objective_components: Path
    elites_mmr_summary: Path
    elites_mmr_sweep: Path
    elites_nn_distance: Path
    chain_trajectory_points: Path
    chain_trajectory_lines: Path

    def to_mapping(self) -> dict[str, Path]:
        return {
            "scores_summary": self.scores_summary,
            "elites_topk": self.elites_topk,
            "metrics_joint": self.metrics_joint,
            "chain_trajectory_points": self.chain_trajectory_points,
            "chain_trajectory_lines": self.chain_trajectory_lines,
            "overlap_pair_summary": self.overlap_pair_summary,
            "overlap_per_elite": self.overlap_per_elite,
            "diagnostics_summary": self.diagnostics_summary,
            "objective_components": self.objective_components,
            "elites_mmr_summary": self.elites_mmr_summary,
            "elites_mmr_sweep": self.elites_mmr_sweep,
            "elites_nn_distance": self.elites_nn_distance,
        }


def _build_table_paths(*, tmp_root: Path, table_ext: str) -> _AnalysisTablePaths:
    return _AnalysisTablePaths(
        scores_summary=analysis_table_path(tmp_root, "scores_summary", table_ext),
        elites_topk=analysis_table_path(tmp_root, "elites_topk", table_ext),
        metrics_joint=analysis_table_path(tmp_root, "metrics_joint", table_ext),
        overlap_pair_summary=analysis_table_path(tmp_root, "overlap_pair_summary", table_ext),
        overlap_per_elite=analysis_table_path(tmp_root, "overlap_per_elite", table_ext),
        diagnostics_summary=analysis_table_path(tmp_root, "diagnostics_summary", "json"),
        objective_components=analysis_table_path(tmp_root, "objective_components", "json"),
        elites_mmr_summary=analysis_table_path(tmp_root, "elites_mmr_summary", table_ext),
        elites_mmr_sweep=analysis_table_path(tmp_root, "elites_mmr_sweep", table_ext),
        elites_nn_distance=analysis_table_path(tmp_root, "elites_nn_distance", table_ext),
        chain_trajectory_points=analysis_table_path(tmp_root, "chain_trajectory_points", table_ext),
        chain_trajectory_lines=analysis_table_path(tmp_root, "chain_trajectory_lines", table_ext),
    )


def _write_score_summary_tables(
    *,
    sequences_df: pd.DataFrame,
    elites_df: pd.DataFrame,
    tf_names: list[str],
    sample_meta: SampleMeta,
    table_paths: _AnalysisTablePaths,
) -> None:
    from dnadesign.cruncher.analysis.plots.summary import (
        score_frame_from_df,
        write_elite_topk,
        write_joint_metrics,
        write_score_summary,
    )

    score_df = score_frame_from_df(sequences_df, tf_names)
    write_score_summary(score_df=score_df, tf_names=tf_names, out_path=table_paths.scores_summary)
    top_k = sample_meta.top_k if sample_meta.top_k else len(elites_df)
    write_elite_topk(elites_df=elites_df, tf_names=tf_names, out_path=table_paths.elites_topk, top_k=top_k)
    write_joint_metrics(elites_df=elites_df, tf_names=tf_names, out_path=table_paths.metrics_joint)


def _compute_trajectory_tables(
    *,
    tmp_root: Path,
    sequences_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    elites_df: pd.DataFrame,
    tf_names: list[str],
    pwms: dict[str, Any],
    analysis_cfg: object,
    objective_from_manifest: dict[str, object],
    bidirectional: bool,
    pwm_pseudocounts: object,
    log_odds_clip: object,
    beta_ladder: list[float] | None,
    retain_sequences: bool,
    table_paths: _AnalysisTablePaths,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    _write_table(trajectory_df, table_paths.chain_trajectory_points)
    trajectory_lines_df = build_chain_trajectory_points(
        trajectory_df,
        max_points=analysis_cfg.max_points,
        retain_sequences=retain_sequences,
    )
    _write_table(trajectory_lines_df, table_paths.chain_trajectory_lines)
    return trajectory_df, trajectory_lines_df, baseline_plot_df, elites_plot_df


def _compute_overlap_and_write(
    *,
    elites_df: pd.DataFrame,
    hits_df: pd.DataFrame,
    tf_names: list[str],
    table_paths: _AnalysisTablePaths,
) -> dict[str, object]:
    overlap_summary_df, elite_overlap_df, overlap_summary = compute_overlap_tables(
        elites_df, hits_df, tf_names, include_sequences=False
    )
    _write_table(overlap_summary_df, table_paths.overlap_pair_summary)
    _write_table(elite_overlap_df, table_paths.overlap_per_elite)
    return overlap_summary


def _compute_diagnostics_and_write(
    *,
    trace_idata: object | None,
    sequences_df: pd.DataFrame,
    elites_df: pd.DataFrame,
    hits_df: pd.DataFrame,
    tf_names: list[str],
    manifest: dict[str, object],
    optimizer_stats: dict[str, object] | None,
    sample_meta: SampleMeta,
    overlap_summary: dict[str, object],
    table_paths: _AnalysisTablePaths,
) -> dict[str, object]:
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
    _write_json(table_paths.diagnostics_summary, diagnostics_payload)
    return diagnostics_payload


def _compute_objective_components_and_write(
    *,
    sequences_df: pd.DataFrame,
    tf_names: list[str],
    top_k: int,
    overlap_summary: dict[str, object],
    table_paths: _AnalysisTablePaths,
) -> dict[str, object]:
    objective_components = compute_objective_components(
        sequences_df=sequences_df,
        tf_names=tf_names,
        top_k=top_k,
        overlap_total_bp_median=overlap_summary.get("overlap_total_bp_median"),
    )
    _write_json(table_paths.objective_components, objective_components)
    return objective_components


def _compute_mmr_outputs_and_write(
    *,
    elites_df: pd.DataFrame,
    hits_df: pd.DataFrame,
    sequences_df: pd.DataFrame,
    elites_meta: dict[str, object],
    tf_names: list[str],
    pwms: dict[str, Any],
    bidirectional: bool,
    table_paths: _AnalysisTablePaths,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    elites_mmr_df, nn_df = _summarize_elites_mmr(
        elites_df,
        hits_df,
        sequences_df,
        elites_meta,
        tf_names,
        pwms,
        bidirectional=bidirectional,
    )
    _write_table(elites_mmr_df, table_paths.elites_mmr_summary)
    _write_table(nn_df, table_paths.elites_nn_distance)
    return elites_mmr_df, nn_df


def _run_optional_mmr_sweep(
    *,
    run_dir: Path,
    analysis_cfg: object,
    table_paths: _AnalysisTablePaths,
) -> None:
    if not analysis_cfg.mmr_sweep.enabled:
        return
    run_mmr_sweep_for_run(
        run_dir,
        pool_size_values=analysis_cfg.mmr_sweep.pool_size_values,
        diversity_values=analysis_cfg.mmr_sweep.diversity_values,
        out_path=table_paths.elites_mmr_sweep,
    )


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
    table_paths = _build_table_paths(tmp_root=tmp_root, table_ext=analysis_cfg.table_format)
    _write_score_summary_tables(
        sequences_df=sequences_df,
        elites_df=elites_df,
        tf_names=tf_names,
        sample_meta=sample_meta,
        table_paths=table_paths,
    )

    objective_from_manifest, bidirectional, pwm_pseudocounts, log_odds_clip, retain_sequences, beta_ladder = (
        _resolve_objective_projection_inputs(
            manifest=manifest,
            sample_meta=sample_meta,
            used_cfg=used_cfg,
            elites_df=elites_df,
            optimizer_stats=optimizer_stats,
        )
    )
    trajectory_df, trajectory_lines_df, baseline_plot_df, elites_plot_df = _compute_trajectory_tables(
        tmp_root=tmp_root,
        sequences_df=sequences_df,
        baseline_df=baseline_df,
        elites_df=elites_df,
        tf_names=tf_names,
        pwms=pwms,
        analysis_cfg=analysis_cfg,
        objective_from_manifest=objective_from_manifest,
        bidirectional=bidirectional,
        pwm_pseudocounts=pwm_pseudocounts,
        log_odds_clip=log_odds_clip,
        beta_ladder=beta_ladder,
        retain_sequences=retain_sequences,
        table_paths=table_paths,
    )
    overlap_summary = _compute_overlap_and_write(
        elites_df=elites_df,
        hits_df=hits_df,
        tf_names=tf_names,
        table_paths=table_paths,
    )
    diagnostics_payload = _compute_diagnostics_and_write(
        trace_idata=trace_idata,
        sequences_df=sequences_df,
        elites_df=elites_df,
        hits_df=hits_df,
        tf_names=tf_names,
        manifest=manifest,
        optimizer_stats=optimizer_stats,
        sample_meta=sample_meta,
        overlap_summary=overlap_summary,
        table_paths=table_paths,
    )
    objective_components = _compute_objective_components_and_write(
        sequences_df=sequences_df,
        tf_names=tf_names,
        top_k=sample_meta.top_k,
        overlap_summary=overlap_summary,
        table_paths=table_paths,
    )
    _, nn_df = _compute_mmr_outputs_and_write(
        elites_df=elites_df,
        hits_df=hits_df,
        sequences_df=sequences_df,
        elites_meta=elites_meta,
        tf_names=tf_names,
        pwms=pwms,
        bidirectional=bool(sample_meta.bidirectional),
        table_paths=table_paths,
    )
    _run_optional_mmr_sweep(run_dir=run_dir, analysis_cfg=analysis_cfg, table_paths=table_paths)

    baseline_seed = _resolve_baseline_seed(baseline_df)
    baseline_nn = compute_baseline_nn_distances(
        baseline_hits_df,
        tf_names,
        pwms,
        seed=baseline_seed,
    )
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
        table_paths=table_paths.to_mapping(),
    )
