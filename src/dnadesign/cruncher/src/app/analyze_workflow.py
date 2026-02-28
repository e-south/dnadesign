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
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from dnadesign.cruncher.analysis.diagnostics import summarize_sampling_diagnostics
from dnadesign.cruncher.analysis.diversity import (
    compute_baseline_nn_distances,
)
from dnadesign.cruncher.analysis.layout import (
    analysis_root,
    analysis_state_root,
    analysis_table_path,
    analysis_used_path,
)
from dnadesign.cruncher.analysis.mmr_sweep_service import run_mmr_sweep_for_run
from dnadesign.cruncher.analysis.objective import compute_objective_components
from dnadesign.cruncher.analysis.overlap import compute_overlap_tables
from dnadesign.cruncher.analysis.trajectory import (
    build_chain_trajectory_points,
)
from dnadesign.cruncher.app.analyze.archive import _prune_latest_analysis_artifacts
from dnadesign.cruncher.app.analyze.metadata import (
    _analysis_id,
    _get_version,
    _load_pwms_from_config,
    _resolve_sample_meta,
)
from dnadesign.cruncher.app.analyze.optimizer_stats import _resolve_optimizer_stats
from dnadesign.cruncher.app.analyze.plan import resolve_analysis_plan
from dnadesign.cruncher.app.analyze.plotting import (
    _prepare_analysis_plot_dir,
    _render_fimo_analysis_plot,
    _render_static_analysis_plots,
    _render_trajectory_analysis_plots,
    _render_trajectory_video_plot,
)
from dnadesign.cruncher.app.analyze.publish import publish_analysis_outputs
from dnadesign.cruncher.app.analyze.run_resolution import _resolve_run_dir, _resolve_run_names
from dnadesign.cruncher.app.analyze.staging import (
    analysis_managed_paths,
    analyze_lock_meta_path,
    delete_path,
    finalize_analysis_root,
    prepare_analysis_root,
    recoverable_analyze_lock_reason,
)
from dnadesign.cruncher.app.analyze_score_space import (
    _project_trajectory_views_with_cleanup,
    _resolve_objective_projection_inputs,
    _resolve_score_space_context,
)
from dnadesign.cruncher.app.analyze_support import (
    _load_elites_meta,
    _load_run_artifacts_for_analysis,
    _resolve_baseline_seed,
    _resolve_tf_names,
    _summarize_elites_mmr,
    _write_analysis_used,
    _write_json,
    _write_table,
)
from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json
from dnadesign.cruncher.artifacts.entries import append_artifacts
from dnadesign.cruncher.artifacts.manifest import load_manifest, write_manifest
from dnadesign.cruncher.config.schema_v3 import CruncherConfig
from dnadesign.cruncher.integrations.meme_suite import resolve_tool_path
from dnadesign.cruncher.utils.arviz_cache import ensure_arviz_data_dir
from dnadesign.cruncher.utils.hashing import sha256_path
from dnadesign.cruncher.utils.paths import resolve_catalog_root
from dnadesign.cruncher.viz.mpl import ensure_mpl_cache

logger = logging.getLogger(__name__)
_ANALYZE_SUPPORT_EXPORTS = (_load_elites_meta,)

__all__ = ["run_analyze"]


def _verify_manifest_lockfile(manifest: dict[str, object]) -> None:
    lockfile_path = manifest.get("lockfile_path")
    lockfile_sha = manifest.get("lockfile_sha256")
    if not lockfile_path or not lockfile_sha:
        return
    lock_path = Path(str(lockfile_path))
    if not lock_path.exists():
        raise FileNotFoundError(f"Lockfile referenced by run manifest missing: {lock_path}")
    current_sha = sha256_path(lock_path)
    if str(current_sha) != str(lockfile_sha):
        raise ValueError("Lockfile checksum mismatch (run manifest does not match current lockfile).")


def _create_analyze_tmp_root(
    *,
    analysis_root_path: Path,
    run_name: str,
    analysis_id: str,
    created_at: str,
) -> Path:
    tmp_root = analysis_state_root(analysis_root_path) / "tmp"
    if tmp_root.exists():
        recoverable_reason = recoverable_analyze_lock_reason(tmp_root)
        if recoverable_reason is not None:
            logger.warning(
                "Recovering stale analyze lock for run '%s' at %s (%s).",
                run_name,
                tmp_root,
                recoverable_reason,
            )
            shutil.rmtree(tmp_root, ignore_errors=True)
        else:
            raise RuntimeError(
                f"Analyze already in progress for run '{run_name}' (lock: {tmp_root}). "
                "If no analyze is running, remove the stale analysis temp directory."
            )
    try:
        tmp_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Analyze already in progress for run '{run_name}' (lock: {tmp_root}). "
            "If no analyze is running, remove the stale analysis temp directory."
        ) from exc
    atomic_write_json(
        analyze_lock_meta_path(tmp_root),
        {
            "analysis_id": analysis_id,
            "run": run_name,
            "created_at": created_at,
            "pid": os.getpid(),
        },
    )
    return tmp_root


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
    cfg: CruncherConfig,
    config_path: Path,
    analysis_cfg: object,
    run_name: str,
    resolved_meme_tool_path: object,
    plot_elites_nn_distance: object,
    plot_elites_showcase: object,
    plot_optimizer_vs_fimo: object,
    plot_health_panel: object,
    plot_chain_trajectory_sweep: object,
    plot_elite_score_space_context: object,
) -> Path:
    run_dir = _resolve_run_dir(cfg, config_path, run_name)
    manifest = load_manifest(run_dir)
    optimizer_stats = _resolve_optimizer_stats(manifest, run_dir)
    _verify_manifest_lockfile(manifest)
    pwms, used_cfg = _load_pwms_from_config(run_dir)
    tf_names = _resolve_tf_names(used_cfg, pwms)
    sample_meta = _resolve_sample_meta(used_cfg, manifest)
    analysis_id = _analysis_id()
    created_at = datetime.now(timezone.utc).isoformat()

    analysis_root_path = analysis_root(run_dir)
    tmp_root = _create_analyze_tmp_root(
        analysis_root_path=analysis_root_path,
        run_name=run_name,
        analysis_id=analysis_id,
        created_at=created_at,
    )

    analysis_used_file = analysis_used_path(tmp_root)

    require_random_baseline = bool(cfg.sample is not None and cfg.sample.output.save_random_baseline)
    artifacts = _load_run_artifacts_for_analysis(
        run_dir,
        require_random_baseline=require_random_baseline,
    )
    sequences_df = artifacts.sequences_df
    elites_df = artifacts.elites_df
    hits_df = artifacts.hits_df
    baseline_df = artifacts.baseline_df
    baseline_hits_df = artifacts.baseline_hits_df
    trace_idata = artifacts.trace_idata
    elites_meta = artifacts.elites_meta

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

    objective_from_manifest, bidirectional, pseudocounts_raw, log_odds_clip_raw, retain_sequences, beta_ladder = (
        _resolve_objective_projection_inputs(
            manifest=manifest,
            sample_meta=sample_meta,
            used_cfg=used_cfg,
            elites_df=elites_df,
            optimizer_stats=optimizer_stats if isinstance(optimizer_stats, dict) else None,
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
        pseudocounts_raw=pseudocounts_raw,
        log_odds_clip_raw=log_odds_clip_raw,
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

    plot_format = analysis_cfg.plot_format
    plot_dpi = analysis_cfg.plot_dpi
    plot_kwargs = {"dpi": plot_dpi, "png_compress_level": 9}
    _prepare_analysis_plot_dir(run_dir)

    score_space_ctx = _resolve_score_space_context(
        tf_names=tf_names,
        analysis_cfg=analysis_cfg,
        elites_plot_df=elites_plot_df,
        pwms=pwms,
        sequences_df=sequences_df,
        manifest=manifest,
        objective_from_manifest=objective_from_manifest,
    )

    _write_analysis_used(
        analysis_used_file,
        analysis_cfg.model_dump(mode="json"),
        analysis_id,
        run_name,
        extras={
            "tf_pair_mode": analysis_cfg.pairwise,
            "tf_pair_selected": list(score_space_ctx.focus_pair) if score_space_ctx.focus_pair else None,
            "trajectory_tf_pair": list(score_space_ctx.trajectory_tf_pair),
            "score_space_mode": score_space_ctx.mode,
            "score_space_pairs": [list(pair) for pair in score_space_ctx.pairs],
            "trajectory_scatter_scale": score_space_ctx.trajectory_scale,
            "trajectory_scatter_retain_elites": analysis_cfg.trajectory_scatter_retain_elites,
            "trajectory_sweep_y_column": analysis_cfg.trajectory_sweep_y_column,
            "trajectory_sweep_mode": analysis_cfg.trajectory_sweep_mode,
            "trajectory_summary_overlay": analysis_cfg.trajectory_summary_overlay,
            "trajectory_video_enabled": analysis_cfg.trajectory_video.enabled,
            "trajectory_video_timeline_mode": analysis_cfg.trajectory_video.timeline_mode,
            "trajectory_video_duration_sec": analysis_cfg.trajectory_video.playback.target_duration_sec,
            "trajectory_video_fps": analysis_cfg.trajectory_video.playback.fps,
            "require_random_baseline": require_random_baseline,
        },
    )

    plot_entries: list[dict[str, object]] = []
    plot_artifacts: list[dict[str, object]] = []
    _render_trajectory_analysis_plots(
        run_dir=run_dir,
        tmp_root=tmp_root,
        plot_format=plot_format,
        plot_kwargs=plot_kwargs,
        trajectory_lines_df=trajectory_lines_df,
        baseline_plot_df=baseline_plot_df,
        elites_plot_df=elites_plot_df,
        tf_names=tf_names,
        score_space_ctx=score_space_ctx,
        analysis_cfg=analysis_cfg,
        objective_from_manifest=objective_from_manifest,
        optimizer_stats=optimizer_stats if isinstance(optimizer_stats, dict) else None,
        sample_meta=sample_meta,
        plot_entries=plot_entries,
        plot_artifacts=plot_artifacts,
        plot_chain_trajectory_sweep=plot_chain_trajectory_sweep,
        plot_elite_score_space_context=plot_elite_score_space_context,
    )
    _render_trajectory_video_plot(
        run_dir=run_dir,
        tmp_root=tmp_root,
        trajectory_df=trajectory_df,
        tf_names=tf_names,
        pwms=pwms,
        analysis_cfg=analysis_cfg,
        bidirectional=bidirectional,
        pwm_pseudocounts=pseudocounts_raw,
        log_odds_clip=log_odds_clip_raw,
        plot_entries=plot_entries,
        plot_artifacts=plot_artifacts,
    )
    _render_static_analysis_plots(
        run_dir=run_dir,
        tmp_root=tmp_root,
        plot_format=plot_format,
        plot_kwargs=plot_kwargs,
        nn_df=nn_df,
        elites_df=elites_df,
        elites_plot_df=elites_plot_df,
        hits_df=hits_df,
        tf_names=tf_names,
        pwms=pwms,
        baseline_nn=baseline_nn,
        objective_from_manifest=objective_from_manifest,
        trace_idata=trace_idata,
        optimizer_stats=optimizer_stats if isinstance(optimizer_stats, dict) else None,
        analysis_cfg=analysis_cfg,
        plot_entries=plot_entries,
        plot_artifacts=plot_artifacts,
        plot_elites_nn_distance=plot_elites_nn_distance,
        plot_elites_showcase=plot_elites_showcase,
        plot_health_panel=plot_health_panel,
    )
    _render_fimo_analysis_plot(
        run_dir=run_dir,
        tmp_root=tmp_root,
        plot_format=plot_format,
        plot_kwargs=plot_kwargs,
        analysis_cfg=analysis_cfg,
        trajectory_df=trajectory_df,
        tf_names=tf_names,
        pwms=pwms,
        bidirectional=bidirectional,
        resolved_meme_tool_path=resolved_meme_tool_path,
        objective_from_manifest=objective_from_manifest,
        plot_entries=plot_entries,
        plot_artifacts=plot_artifacts,
        plot_optimizer_vs_fimo=plot_optimizer_vs_fimo,
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
    analysis_artifacts = publish_analysis_outputs(
        analysis_id=analysis_id,
        created_at=created_at,
        run_name=run_name,
        analysis_root_path=analysis_root_path,
        tmp_root=tmp_root,
        run_dir=run_dir,
        analysis_used_file=analysis_used_file,
        analysis_cfg_payload=analysis_cfg.model_dump(mode="json"),
        tf_names=tf_names,
        diagnostics_payload=diagnostics_payload,
        objective_components=objective_components,
        overlap_summary=overlap_summary,
        table_paths=table_paths,
        mmr_sweep_enabled=analysis_cfg.mmr_sweep.enabled,
        plot_entries=plot_entries,
        plot_artifacts=plot_artifacts,
        version=_get_version(),
    )

    _finalize_analyze_root_with_recovery(
        analysis_root_path=analysis_root_path,
        analysis_id=analysis_id,
        tmp_root=tmp_root,
        archive=analysis_cfg.archive,
    )

    _prune_latest_analysis_artifacts(manifest)
    append_artifacts(manifest, analysis_artifacts)
    write_manifest(run_dir, manifest)

    return analysis_root_path


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
    resolved_meme_tool_path = resolve_tool_path(cfg.discover.tool_path, config_path=config_path)

    from dnadesign.cruncher.analysis.plots.elites_nn_distance import plot_elites_nn_distance
    from dnadesign.cruncher.analysis.plots.elites_showcase import plot_elites_showcase
    from dnadesign.cruncher.analysis.plots.fimo_concordance import plot_optimizer_vs_fimo
    from dnadesign.cruncher.analysis.plots.health_panel import plot_health_panel
    from dnadesign.cruncher.analysis.plots.trajectory_score_space_plot import plot_elite_score_space_context
    from dnadesign.cruncher.analysis.plots.trajectory_sweep import plot_chain_trajectory_sweep

    results: list[Path] = []
    for run_name in runs:
        results.append(
            _run_analyze_for_run(
                cfg=cfg,
                config_path=config_path,
                analysis_cfg=analysis_cfg,
                run_name=run_name,
                resolved_meme_tool_path=resolved_meme_tool_path,
                plot_elites_nn_distance=plot_elites_nn_distance,
                plot_elites_showcase=plot_elites_showcase,
                plot_optimizer_vs_fimo=plot_optimizer_vs_fimo,
                plot_health_panel=plot_health_panel,
                plot_chain_trajectory_sweep=plot_chain_trajectory_sweep,
                plot_elite_score_space_context=plot_elite_score_space_context,
            )
        )
    return results
