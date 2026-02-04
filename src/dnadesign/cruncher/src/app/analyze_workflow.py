"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/analyze_workflow.py

Analyze sampling runs and produce summary reports.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dnadesign.cruncher.analysis.layout import (
    ANALYSIS_LAYOUT_VERSION,
    analysis_manifest_path,
    analysis_meta_root,
    analysis_used_path,
    plot_manifest_path,
    report_json_path,
    report_md_path,
    summary_path,
    table_manifest_path,
)
from dnadesign.cruncher.analysis.objective import compute_objective_components
from dnadesign.cruncher.analysis.overlap import compute_overlap_tables
from dnadesign.cruncher.analysis.parquet import read_parquet, write_parquet
from dnadesign.cruncher.analysis.plot_registry import PLOT_SPECS
from dnadesign.cruncher.analysis.report import ensure_report
from dnadesign.cruncher.app.analyze.archive import (
    _analysis_item_paths,
    _analysis_signature,
    _archive_existing_analysis,
    _clear_latest_analysis,
    _load_summary_id,
    _load_summary_payload,
    _prune_latest_analysis_artifacts,
    _rewrite_manifest_paths,
    _update_archived_summary,
)
from dnadesign.cruncher.app.analyze.diagnostics import _summarize_move_stats
from dnadesign.cruncher.app.analyze.metadata import (
    SampleMeta,
    _analysis_id,
    _auto_select_tf_pair,
    _get_git_commit,
    _get_version,
    _load_pwms_from_config,
    _resolve_git_dir,
    _resolve_sample_meta,
    _resolve_scoring_params,
    _resolve_tf_pair,
)
from dnadesign.cruncher.app.run_service import list_runs
from dnadesign.cruncher.artifacts.entries import (
    append_artifacts,
    artifact_entry,
)
from dnadesign.cruncher.artifacts.layout import (
    config_used_path,
    elites_yaml_path,
    sequences_path,
    trace_path,
)
from dnadesign.cruncher.artifacts.manifest import load_manifest
from dnadesign.cruncher.config.schema_v2 import CruncherConfig
from dnadesign.cruncher.utils.hashing import sha256_path
from dnadesign.cruncher.utils.paths import resolve_catalog_root
from dnadesign.cruncher.viz.mpl import ensure_mpl_cache

logger = logging.getLogger(__name__)

__all__ = [
    "SampleMeta",
    "_analysis_id",
    "_analysis_item_paths",
    "_analysis_signature",
    "_archive_existing_analysis",
    "_auto_select_tf_pair",
    "_clear_latest_analysis",
    "_get_git_commit",
    "_get_version",
    "_load_pwms_from_config",
    "_load_summary_id",
    "_load_summary_payload",
    "_prune_latest_analysis_artifacts",
    "_resolve_git_dir",
    "_resolve_sample_meta",
    "_resolve_scoring_params",
    "_resolve_tf_pair",
    "_rewrite_manifest_paths",
    "_summarize_move_stats",
    "_update_archived_summary",
    "run_analyze",
]


def run_analyze(
    cfg: CruncherConfig,
    config_path: Path,
    *,
    runs_override: list[str] | None = None,
    use_latest: bool = False,
    tf_pair_override: tuple[str, str] | None = None,
    plot_keys_override: list[str] | None = None,
    scatter_background_override: bool | None = None,
    scatter_background_samples_override: int | None = None,
    scatter_background_seed_override: int | None = None,
) -> list[Path]:
    """Iterate over runs listed in cfg.analysis.runs and generate diagnostics."""
    # Shared PWM store
    if cfg.analysis is None:
        raise ValueError("analysis section is required for analyze")
    if cfg.sample is None:
        raise ValueError("sample section is required for analyze")

    ensure_mpl_cache(resolve_catalog_root(config_path, cfg.motif_store.catalog_root))
    import arviz as az

    from dnadesign.cruncher.analysis.diagnostics import summarize_sampling_diagnostics
    from dnadesign.cruncher.analysis.per_pwm import gather_per_pwm_scores
    from dnadesign.cruncher.analysis.plots._savefig import savefig
    from dnadesign.cruncher.analysis.plots.dashboard import plot_dashboard
    from dnadesign.cruncher.analysis.plots.diagnostics import (
        make_pair_idata,
        plot_autocorr,
        plot_ess,
        plot_pair_pwm_scores,
        plot_parallel_pwm_scores,
        plot_rank_diagnostic,
        plot_trace,
        report_convergence,
    )
    from dnadesign.cruncher.analysis.plots.moves import (
        plot_move_acceptance_time,
        plot_move_usage_time,
        plot_pt_swap_by_pair,
    )
    from dnadesign.cruncher.analysis.plots.optimization import (
        plot_elite_filter_waterfall,
        plot_worst_tf_identity,
        plot_worst_tf_trace,
    )
    from dnadesign.cruncher.analysis.plots.overlap import (
        plot_motif_offset_rug,
        plot_overlap_bp_distribution,
        plot_overlap_heatmap,
        plot_overlap_strand_combos,
    )
    from dnadesign.cruncher.analysis.plots.placeholders import (
        write_placeholder_plot,
        write_placeholder_text,
    )
    from dnadesign.cruncher.analysis.plots.scatter import plot_scatter
    from dnadesign.cruncher.analysis.plots.summary import (
        plot_correlation_heatmap,
        plot_parallel_coords,
        plot_score_box,
        plot_score_hist,
        plot_score_pairgrid,
        score_frame_from_df,
        write_elite_topk,
        write_joint_metrics,
        write_score_summary,
    )

    analysis_cfg = cfg.analysis
    plot_keys = {spec.key for spec in PLOT_SPECS}
    tier0_plot_keys = {
        "dashboard",
        "worst_tf_trace",
        "worst_tf_identity",
        "elite_filter_waterfall",
        "overlap_heatmap",
        "overlap_bp_distribution",
    }
    mcmc_plot_keys = {
        "trace",
        "autocorr",
        "convergence",
        "pt_swap_by_pair",
        "move_acceptance_time",
        "move_usage_time",
    }
    extra_plot_keys = plot_keys - tier0_plot_keys - mcmc_plot_keys
    extra_plots = analysis_cfg.extra_plots
    mcmc_diagnostics = analysis_cfg.mcmc_diagnostics
    override_payload: dict[str, object] = {}
    auto_adjustments: dict[str, object] = {}
    if plot_keys_override:
        requested = [key for key in plot_keys_override if key]
        if "all" in requested and len(requested) > 1:
            raise ValueError("Use either --plots all or explicit plot keys, not both.")
        unknown = [key for key in requested if key != "all" and key not in plot_keys]
        if unknown:
            raise ValueError(f"Unknown plot keys: {', '.join(unknown)}")
        analysis_cfg = analysis_cfg.model_copy(deep=True)
        for key in plot_keys:
            setattr(analysis_cfg.plots, key, False)
        if "all" in requested:
            for key in plot_keys:
                setattr(analysis_cfg.plots, key, True)
            extra_plots = True
            mcmc_diagnostics = True
            override_payload["extra_plots"] = True
            override_payload["mcmc_diagnostics"] = True
        else:
            for key in requested:
                setattr(analysis_cfg.plots, key, True)
            if any(key in extra_plot_keys for key in requested):
                extra_plots = True
                override_payload["extra_plots"] = True
            if any(key in mcmc_plot_keys for key in requested):
                mcmc_diagnostics = True
                override_payload["mcmc_diagnostics"] = True
        override_payload["plots"] = requested

    if scatter_background_override is not None or scatter_background_samples_override is not None:
        if analysis_cfg is cfg.analysis:
            analysis_cfg = analysis_cfg.model_copy(deep=True)
        if scatter_background_override is not None:
            analysis_cfg.scatter_background = scatter_background_override
            override_payload["scatter_background"] = scatter_background_override
        if scatter_background_samples_override is not None:
            if scatter_background_samples_override < 0:
                raise ValueError("--scatter-background-samples must be >= 0.")
            analysis_cfg.scatter_background_samples = scatter_background_samples_override
            override_payload["scatter_background_samples"] = scatter_background_samples_override

    explicit_extra = [key for key in extra_plot_keys if getattr(analysis_cfg.plots, key, False)]
    if explicit_extra and not analysis_cfg.extra_plots:
        if analysis_cfg is cfg.analysis:
            analysis_cfg = analysis_cfg.model_copy(deep=True)
        analysis_cfg.extra_plots = True
        extra_plots = True
        auto_adjustments["extra_plots_forced"] = explicit_extra
        logger.info(
            "Extra plots enabled in config; forcing analysis.extra_plots=true for: %s",
            ", ".join(explicit_extra),
        )
    explicit_mcmc = [key for key in mcmc_plot_keys if getattr(analysis_cfg.plots, key, False)]
    if explicit_mcmc and not analysis_cfg.mcmc_diagnostics:
        if analysis_cfg is cfg.analysis:
            analysis_cfg = analysis_cfg.model_copy(deep=True)
        analysis_cfg.mcmc_diagnostics = True
        mcmc_diagnostics = True
        auto_adjustments["mcmc_diagnostics_forced"] = explicit_mcmc
        logger.info(
            "MCMC diagnostics plots enabled in config; forcing analysis.mcmc_diagnostics=true for: %s",
            ", ".join(explicit_mcmc),
        )
    if scatter_background_seed_override is not None:
        if scatter_background_seed_override < 0:
            raise ValueError("--scatter-background-seed must be >= 0.")
        if analysis_cfg is cfg.analysis:
            analysis_cfg = analysis_cfg.model_copy(deep=True)
        analysis_cfg.scatter_background_seed = scatter_background_seed_override
        override_payload["scatter_background_seed"] = scatter_background_seed_override

    if analysis_cfg.extra_plots != extra_plots or analysis_cfg.mcmc_diagnostics != mcmc_diagnostics:
        if analysis_cfg is cfg.analysis:
            analysis_cfg = analysis_cfg.model_copy(deep=True)
        analysis_cfg.extra_plots = extra_plots
        analysis_cfg.mcmc_diagnostics = mcmc_diagnostics

    disabled_plots: list[str] = []
    if not analysis_cfg.extra_plots:
        if analysis_cfg is cfg.analysis:
            analysis_cfg = analysis_cfg.model_copy(deep=True)
        for key in sorted(extra_plot_keys):
            if getattr(analysis_cfg.plots, key, False):
                setattr(analysis_cfg.plots, key, False)
                disabled_plots.append(key)
        if disabled_plots:
            logger.info(
                "Disabled extra plots (analysis.extra_plots=false): %s",
                ", ".join(disabled_plots),
            )
    disabled_mcmc_plots: list[str] = []
    if not analysis_cfg.mcmc_diagnostics:
        if analysis_cfg is cfg.analysis:
            analysis_cfg = analysis_cfg.model_copy(deep=True)
        for key in sorted(mcmc_plot_keys):
            if getattr(analysis_cfg.plots, key, False):
                setattr(analysis_cfg.plots, key, False)
                disabled_mcmc_plots.append(key)
        if disabled_mcmc_plots:
            logger.info(
                "Disabled MCMC diagnostics plots (analysis.mcmc_diagnostics=false): %s",
                ", ".join(disabled_mcmc_plots),
            )
    explicit_overrides = set(plot_keys_override or [])
    override_all = "all" in explicit_overrides
    if analysis_cfg.dashboard_only and analysis_cfg.plots.dashboard:
        dashboard_components = {
            "worst_tf_trace",
            "worst_tf_identity",
            "overlap_heatmap",
            "overlap_bp_distribution",
        }
        disabled_dashboard: list[str] = []
        if analysis_cfg is cfg.analysis:
            analysis_cfg = analysis_cfg.model_copy(deep=True)
        for key in sorted(dashboard_components):
            if override_all or key in explicit_overrides:
                continue
            if getattr(analysis_cfg.plots, key, False):
                setattr(analysis_cfg.plots, key, False)
                disabled_dashboard.append(key)
        if disabled_dashboard:
            auto_adjustments["dashboard_only_disabled"] = disabled_dashboard
            logger.info(
                "Dashboard-only enabled; disabled redundant plots: %s",
                ", ".join(disabled_dashboard),
            )

    override_payload = override_payload or None

    cfg_effective = cfg
    if analysis_cfg is not cfg.analysis:
        cfg_effective = cfg.model_copy(deep=True)
        cfg_effective.analysis = analysis_cfg
    plot_dpi = analysis_cfg.plot_dpi
    png_compress_level = analysis_cfg.png_compress_level
    plot_format = analysis_cfg.plot_format

    runs = runs_override if runs_override else (analysis_cfg.runs or [])
    if not runs:
        if not use_latest:
            logger.info("No analysis runs configured; defaulting to latest sample run.")
            use_latest = True
        latest = list_runs(cfg, config_path, stage="sample")
        if not latest:
            raise ValueError("No sample runs found. Run `cruncher sample <config>` first.")
        runs = [latest[0].name]
        logger.info("Analyzing latest sample run: %s", runs[0])
    analysis_runs: list[Path] = []
    from dnadesign.cruncher.app.run_service import get_run

    for run_name in runs:
        run_info = get_run(cfg, config_path, run_name)
        sample_dir = run_info.run_dir

        if run_info.stage != "sample":
            raise ValueError(f"Run '{run_name}' is not a sample run (stage={run_info.stage})")

        logger.info("Analyzing run: %s", run_info.name)
        if run_name != run_info.name:
            logger.debug("Run path override: %s", run_name)
        manifest = load_manifest(sample_dir)
        if manifest.get("stage") != "sample":
            raise ValueError(f"Run '{run_name}' is not a sample run (stage={manifest.get('stage')})")
        lock_path_raw = manifest.get("lockfile_path")
        if not lock_path_raw:
            raise ValueError("Run manifest missing lockfile_path; re-run `cruncher sample`.")
        lock_path = Path(lock_path_raw)
        if not lock_path.exists():
            raise FileNotFoundError(f"Lockfile not found: {lock_path}")
        expected_sha = manifest.get("lockfile_sha256")
        if expected_sha:
            actual_sha = sha256_path(lock_path)
            if actual_sha != expected_sha:
                raise ValueError(f"Lockfile checksum mismatch for {lock_path}")

        # ── reconstruct PWMs from config_used.yaml for reproducibility ───────
        pwms, used_cfg = _load_pwms_from_config(sample_dir)
        tf_names = list(pwms.keys())
        sample_meta = _resolve_sample_meta(cfg, used_cfg)
        scoring_pseudocounts, scoring_log_odds_clip = _resolve_scoring_params(used_cfg)
        bidirectional = sample_meta.bidirectional
        used_sample = used_cfg.get("sample") if isinstance(used_cfg, dict) else None
        if isinstance(used_sample, dict):
            trace_cfg = used_sample.get("output", {}).get("trace", {})
            if isinstance(trace_cfg, dict) and trace_cfg.get("include_tune"):
                logger.warning(
                    "sample.output.trace.include_tune affects sequences.parquet only; trace.nc contains draw samples."
                )

        # ── locate elites parquet file ──────────────────────────────────────
        from dnadesign.cruncher.analysis.elites import find_elites_parquet

        elites_path = find_elites_parquet(sample_dir)
        elites_df = read_parquet(elites_path)
        logger.info(
            "Loaded %d elites from %s",
            len(elites_df),
            elites_path.relative_to(sample_dir),
        )
        elites_meta: dict[str, object] = {}
        elites_meta_path = elites_yaml_path(sample_dir)
        if elites_meta_path.exists():
            try:
                elites_meta = yaml.safe_load(elites_meta_path.read_text()) or {}
            except Exception as exc:
                logger.warning("Failed to read elites metadata (%s): %s", elites_meta_path, exc)

        # ── load trace & sequences for downstream plots ───────────────────────
        seq_path, trace_file = sequences_path(sample_dir), trace_path(sample_dir)
        if not seq_path.exists():
            raise FileNotFoundError(
                f"Missing artifacts/sequences.parquet in {sample_dir}. "
                "Re-run `cruncher sample` with sample.output.save_sequences=true."
            )
        plots = analysis_cfg.plots
        needs_trace = analysis_cfg.mcmc_diagnostics or plots.trace or plots.autocorr or plots.convergence
        trace_idata = None
        if trace_file.exists() and needs_trace:
            trace_idata = az.from_netcdf(trace_file)
        elif needs_trace:
            logger.warning(
                "Missing artifacts/trace.nc in %s; trace-based diagnostics will be skipped.",
                sample_dir,
            )

        analysis_root = sample_dir / "analysis"
        analysis_root.mkdir(parents=True, exist_ok=True)
        analysis_signature, signature_payload = _analysis_signature(
            analysis_cfg=analysis_cfg,
            override_payload=override_payload,
            config_used_file=config_used_path(sample_dir),
            sequences_file=seq_path,
            elites_file=elites_path,
            trace_file=trace_file,
        )
        existing_summary = _load_summary_payload(analysis_root)
        if existing_summary and existing_summary.get("signature") == analysis_signature:
            logger.info("Analysis already up to date: %s", analysis_root)
            report_json = report_json_path(analysis_root)
            report_md = report_md_path(analysis_root)
            if not report_json.exists() or not report_md.exists():
                analysis_used_payload = None
                analysis_used_file = analysis_used_path(analysis_root)
                if analysis_used_file.exists():
                    try:
                        analysis_used_payload = yaml.safe_load(analysis_used_file.read_text()) or {}
                    except Exception as exc:
                        logger.warning("Failed to read %s: %s", analysis_used_file, exc)
                ensure_report(
                    analysis_root=analysis_root,
                    summary_payload=existing_summary,
                    analysis_used_payload=analysis_used_payload,
                )
                summary_file = summary_path(analysis_root)
                if summary_file.exists():
                    if "report_json" not in existing_summary or "report_md" not in existing_summary:
                        existing_summary["report_json"] = str(report_json.relative_to(sample_dir))
                        existing_summary["report_md"] = str(report_md.relative_to(sample_dir))
                        summary_file.write_text(json.dumps(existing_summary, indent=2))
                analysis_manifest_file = analysis_manifest_path(analysis_root)
                if analysis_manifest_file.exists():
                    manifest_payload = json.loads(analysis_manifest_file.read_text())
                    artifacts = manifest_payload.get("artifacts") or []
                    report_entries = [
                        {
                            "path": str(report_json.relative_to(sample_dir)),
                            "kind": "report",
                            "label": "Analysis report (JSON)",
                            "reason": "default",
                            "exists": report_json.exists(),
                            "key": "report_json",
                        },
                        {
                            "path": str(report_md.relative_to(sample_dir)),
                            "kind": "report",
                            "label": "Analysis report (Markdown)",
                            "reason": "default",
                            "exists": report_md.exists(),
                            "key": "report_md",
                        },
                    ]
                    seen = {item.get("path") for item in artifacts if isinstance(item, dict)}
                    for entry in report_entries:
                        if entry["path"] in seen:
                            continue
                        artifacts.append(entry)
                    manifest_payload["artifacts"] = artifacts
                    analysis_manifest_file.write_text(json.dumps(manifest_payload, indent=2))
                report_artifacts = [
                    artifact_entry(
                        report_json,
                        sample_dir,
                        kind="json",
                        label="Analysis report (JSON)",
                        stage="analysis",
                    ),
                    artifact_entry(
                        report_md,
                        sample_dir,
                        kind="text",
                        label="Analysis report (Markdown)",
                        stage="analysis",
                    ),
                ]
                append_artifacts(manifest, report_artifacts)
                from dnadesign.cruncher.app.run_service import update_run_index_from_manifest
                from dnadesign.cruncher.artifacts.manifest import write_manifest

                write_manifest(sample_dir, manifest)
                update_run_index_from_manifest(
                    config_path,
                    sample_dir,
                    manifest,
                    catalog_root=cfg.motif_store.catalog_root,
                )
            analysis_runs.append(analysis_root)
            continue
        analysis_id = _analysis_id()
        existing_items = [path for path in _analysis_item_paths(analysis_root) if path.exists()]
        existing_id = _load_summary_id(analysis_root)
        if existing_items and existing_id is None:
            raise ValueError(
                "analysis/ contains artifacts but summary.json is missing. "
                "Remove analysis/ or restore summary.json before re-running analyze."
            )
        if existing_id:
            if analysis_cfg.archive:
                _archive_existing_analysis(analysis_root, manifest, existing_id)
            else:
                _clear_latest_analysis(analysis_root)
                _prune_latest_analysis_artifacts(manifest)
        analysis_meta = analysis_meta_root(analysis_root)
        analysis_meta.mkdir(parents=True, exist_ok=True)
        plots_dir = analysis_root
        tables_dir = analysis_root

        def _plot_path(stem: str) -> Path:
            return plots_dir / f"{stem}.{plot_format}"

        analysis_used_file = analysis_used_path(analysis_root)
        analysis_used_payload = {
            "analysis_id": analysis_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "layout_version": ANALYSIS_LAYOUT_VERSION,
            "analysis": analysis_cfg.model_dump(),
        }
        if override_payload:
            analysis_used_payload["analysis_overrides"] = override_payload
            analysis_used_payload["analysis_base"] = cfg.analysis.model_dump()

        seq_df = read_parquet(seq_path)
        score_df = score_frame_from_df(seq_df, tf_names)

        # gather per-PWM scores (scatter plot only)
        per_pwm_path: Path | None = None
        if plots.scatter_pwm:
            per_pwm_path = tables_dir / f"per_pwm_scores.{analysis_cfg.table_format}"
            gather_per_pwm_scores(
                sample_dir,
                analysis_cfg.subsampling_epsilon,
                pwms,
                bidirectional=bidirectional,
                scale=analysis_cfg.scatter_scale,
                out_path=per_pwm_path,
                sequences_df=seq_df,
                pseudocounts=scoring_pseudocounts,
                log_odds_clip=scoring_log_odds_clip,
            )
            logger.info("Wrote per-PWM score table → %s", per_pwm_path.relative_to(sample_dir))
        table_ext = analysis_cfg.table_format
        score_summary_path = tables_dir / f"score_summary.{table_ext}"
        write_score_summary(score_df, tf_names, score_summary_path)

        topk_path = tables_dir / f"elite_topk.{table_ext}"
        write_elite_topk(elites_df, tf_names, topk_path, top_k=sample_meta.top_k)

        joint_metrics_path = tables_dir / f"joint_metrics.{table_ext}"
        write_joint_metrics(elites_df, tf_names, joint_metrics_path)

        pwm_widths = {tf: pwm.length for tf, pwm in pwms.items()}
        overlap_summary_df, elite_overlap_df, overlap_summary = compute_overlap_tables(
            elites_df,
            tf_names,
            pwm_widths=pwm_widths,
            include_sequences=analysis_cfg.include_sequences_in_tables,
        )
        overlap_summary_path = tables_dir / f"overlap_summary.{table_ext}"
        elite_overlap_path = tables_dir / f"elite_overlap.{table_ext}"
        if table_ext == "parquet":
            write_parquet(overlap_summary_df, overlap_summary_path)
            write_parquet(elite_overlap_df, elite_overlap_path)
        else:
            overlap_summary_df.to_csv(overlap_summary_path, index=False)
            elite_overlap_df.to_csv(elite_overlap_path, index=False)

        sample_meta_payload = {
            "mode": sample_meta.mode,
            "optimizer_kind": sample_meta.optimizer_kind,
            "chains": sample_meta.chains,
            "draws": sample_meta.draws,
            "tune": sample_meta.tune,
            "top_k": sample_meta.top_k,
            "pwm_sum_threshold": sample_meta.pwm_sum_threshold,
            "bidirectional": sample_meta.bidirectional,
            "move_probs": sample_meta.move_probs,
            "cooling_kind": sample_meta.cooling_kind,
            "dsdna_canonicalize": sample_meta.dsdna_canonicalize,
            "dsdna_hamming": sample_meta.dsdna_hamming,
        }
        diagnostics_summary = summarize_sampling_diagnostics(
            trace_idata=trace_idata,
            sequences_df=seq_df,
            elites_df=elites_df,
            tf_names=tf_names,
            optimizer=manifest.get("optimizer", {}),
            optimizer_stats=manifest.get("optimizer_stats", {}),
            mode=sample_meta.mode,
            optimizer_kind=sample_meta.optimizer_kind,
            sample_meta=sample_meta_payload,
            trace_required=analysis_cfg.mcmc_diagnostics,
            overlap_summary=overlap_summary,
        )
        diagnostics_path = tables_dir / "diagnostics.json"
        diagnostics_path.write_text(json.dumps(diagnostics_summary, indent=2))
        if diagnostics_summary.get("warnings"):
            logger.warning(
                "Diagnostics warnings detected (%d). See %s.",
                len(diagnostics_summary["warnings"]),
                diagnostics_path.relative_to(sample_dir),
            )

        objective_components_path = tables_dir / "objective_components.json"
        objective_components = compute_objective_components(
            seq_df,
            tf_names,
            top_k=sample_meta.top_k,
            dsdna_canonicalize=sample_meta.dsdna_canonicalize,
            overlap_total_bp_median=overlap_summary.get("overlap_total_bp_median"),
        )
        objective_components_path.write_text(json.dumps(objective_components, indent=2))

        move_stats_path = None
        move_stats_df = None
        move_stats_summary_path = None
        move_stats_summary_df = None
        optimizer_stats = manifest.get("optimizer_stats", {})
        move_stats = optimizer_stats.get("move_stats") if isinstance(optimizer_stats, dict) else None
        if isinstance(move_stats, list) and move_stats:
            move_stats_df = pd.DataFrame(move_stats) if analysis_cfg.mcmc_diagnostics else None
            move_stats_summary_df = _summarize_move_stats(move_stats)
            if move_stats_summary_df is not None and not move_stats_summary_df.empty:
                move_stats_summary_path = tables_dir / f"move_stats_summary.{table_ext}"
                if table_ext == "parquet":
                    write_parquet(move_stats_summary_df, move_stats_summary_path)
                else:
                    move_stats_summary_df.to_csv(move_stats_summary_path, index=False)
            if analysis_cfg.extra_tables and move_stats_df is not None:
                move_stats_path = tables_dir / f"move_stats.{table_ext}"
                if table_ext == "parquet":
                    write_parquet(move_stats_df, move_stats_path)
                else:
                    move_stats_df.to_csv(move_stats_path, index=False)

        pt_swap_pairs_path = None
        pt_swap_pairs_df = None
        if analysis_cfg.mcmc_diagnostics and isinstance(optimizer_stats, dict) and sample_meta.optimizer_kind == "pt":
            attempts = optimizer_stats.get("swap_attempts_by_pair")
            accepts = optimizer_stats.get("swap_accepts_by_pair")
            beta_ladder = optimizer_stats.get("beta_ladder_base") or []
            if isinstance(attempts, list) and isinstance(accepts, list):
                rows = []
                for idx, (att, acc) in enumerate(zip(attempts, accepts)):
                    att = int(att)
                    acc = int(acc)
                    rate = acc / float(att) if att else 0.0
                    beta_low = None
                    beta_high = None
                    if idx < len(beta_ladder):
                        beta_low = beta_ladder[idx]
                    if idx + 1 < len(beta_ladder):
                        beta_high = beta_ladder[idx + 1]
                    rows.append(
                        {
                            "pair_index": idx,
                            "beta_low": beta_low,
                            "beta_high": beta_high,
                            "attempts": att,
                            "accepts": acc,
                            "acceptance_rate": rate,
                        }
                    )
                if rows:
                    pt_swap_pairs_df = pd.DataFrame(rows)
                    pt_swap_pairs_path = tables_dir / f"pt_swap_pairs.{table_ext}"
                    if table_ext == "parquet":
                        write_parquet(pt_swap_pairs_df, pt_swap_pairs_path)
                    else:
                        pt_swap_pairs_df.to_csv(pt_swap_pairs_path, index=False)

        auto_opt_table_path = None
        auto_opt_plot_path = None
        auto_opt_payload = manifest.get("auto_opt")
        if isinstance(auto_opt_payload, dict):
            candidates = auto_opt_payload.get("candidates")
            if isinstance(candidates, list) and candidates:
                if analysis_cfg.extra_tables:
                    auto_opt_table_path = tables_dir / f"auto_opt_pilots.{table_ext}"
                    df_auto_table = pd.DataFrame(candidates)
                    if table_ext == "parquet":
                        write_parquet(df_auto_table, auto_opt_table_path)
                    else:
                        df_auto_table.to_csv(auto_opt_table_path, index=False)
                if analysis_cfg.extra_plots:
                    df_auto = pd.DataFrame(candidates)
                    score_col = "top_k_median_final" if "top_k_median_final" in df_auto.columns else "best_score"
                    required_cols = {score_col, "balance_median"}
                    if required_cols.issubset({col for col in df_auto.columns}):
                        df_auto = df_auto.dropna(subset=[score_col, "balance_median"])
                        if not df_auto.empty:
                            import matplotlib.pyplot as plt

                            auto_opt_plot_path = _plot_path("auto_opt_tradeoffs")
                            fig, ax = plt.subplots(figsize=(6, 4))
                            lengths = df_auto.get("length")
                            colors = None
                            if lengths is not None and lengths.notna().any():
                                colors = lengths.astype(float)
                            scatter = ax.scatter(
                                df_auto["balance_median"],
                                df_auto[score_col],
                                c=colors,
                                cmap="viridis" if colors is not None else None,
                                s=50,
                                alpha=0.85,
                            )
                            ax.set_xlabel("Balance (median min normalized)")
                            ax.set_ylabel("Top-K median score (final beta)")
                            ax.set_title("Auto-opt tradeoffs")
                            if colors is not None:
                                fig.colorbar(scatter, ax=ax, label="Length")
                            fig.tight_layout()
                            savefig(fig, auto_opt_plot_path, dpi=plot_dpi, png_compress_level=png_compress_level)
                            plt.close(fig)

        enabled_specs = [spec for spec in PLOT_SPECS if getattr(plots, spec.key, False)]
        if enabled_specs:
            logger.debug("Enabled plots: %s", ", ".join(spec.key for spec in enabled_specs))
        tf_pair = _resolve_tf_pair(analysis_cfg, tf_names, tf_pair_override=tf_pair_override)
        pairwise_requested = any("tf_pair" in spec.requires for spec in enabled_specs)
        autopick_payload = None
        pairwise_placeholder_reason = None
        pairwise_placeholder_keys: list[str] = []
        if pairwise_requested and tf_pair is None:
            tf_pair, reason = _auto_select_tf_pair(score_df, tf_names)
            if tf_pair is not None:
                autopick_payload = {"tf_pair": list(tf_pair), "reason": reason}
                logger.info("Auto-selected tf_pair=%s (%s)", tf_pair, reason)
            else:
                pairwise_placeholder_reason = reason
                pairwise_placeholder_keys = [
                    key
                    for key in ("pair_pwm", "parallel_pwm", "scatter_pwm")
                    if getattr(analysis_cfg.plots, key, False)
                ]
                logger.warning("Pairwise plots will be placeholders: %s", reason)
                tf_pair = None
        if pairwise_placeholder_keys:
            auto_adjustments["pairwise_placeholders"] = {
                "plots": pairwise_placeholder_keys,
                "reason": pairwise_placeholder_reason,
            }

        if autopick_payload:
            analysis_used_payload["analysis_autopicks"] = autopick_payload
        if auto_adjustments:
            analysis_used_payload["analysis_auto_adjustments"] = auto_adjustments
        analysis_used_file.parent.mkdir(parents=True, exist_ok=True)
        analysis_used_file.write_text(yaml.safe_dump(analysis_used_payload, sort_keys=False))

        # ── diagnostics ───────────────────────────────────────────────────────
        if plots.trace:
            if trace_idata is not None:
                plot_trace(
                    trace_idata,
                    plots_dir,
                    dpi=plot_dpi,
                    png_compress_level=png_compress_level,
                    plot_format=plot_format,
                )
            else:
                write_placeholder_plot(
                    _plot_path("diag__trace_score"),
                    "Trace missing: trace-based diagnostics unavailable.",
                    dpi=plot_dpi,
                    png_compress_level=png_compress_level,
                )
        if plots.autocorr:
            if trace_idata is not None:
                plot_autocorr(
                    trace_idata,
                    plots_dir,
                    dpi=plot_dpi,
                    png_compress_level=png_compress_level,
                    plot_format=plot_format,
                )
            else:
                write_placeholder_plot(
                    _plot_path("diag__autocorr_score"),
                    "Trace missing: autocorrelation unavailable.",
                    dpi=plot_dpi,
                    png_compress_level=png_compress_level,
                )
        if plots.convergence:
            idata_for_convergence = trace_idata
            if sample_meta.optimizer_kind == "pt" and trace_idata is not None:
                score = trace_idata.posterior.get("score") if hasattr(trace_idata, "posterior") else None
                if score is not None:
                    values = np.asarray(score)
                    if values.ndim >= 2 and values.shape[0] > 0:
                        idata_for_convergence = az.from_dict(posterior={"score": values[-1:, :]})
            if idata_for_convergence is not None:
                report_convergence(idata_for_convergence, plots_dir)
                plot_rank_diagnostic(
                    idata_for_convergence,
                    plots_dir,
                    dpi=plot_dpi,
                    png_compress_level=png_compress_level,
                    plot_format=plot_format,
                )
                plot_ess(
                    idata_for_convergence,
                    plots_dir,
                    dpi=plot_dpi,
                    png_compress_level=png_compress_level,
                    plot_format=plot_format,
                )
            else:
                write_placeholder_text(plots_dir / "diag__convergence.txt", "Trace missing: convergence unavailable.")
                write_placeholder_plot(
                    _plot_path("diag__rank_plot_score"),
                    "Trace missing: rank plot unavailable.",
                    dpi=plot_dpi,
                    png_compress_level=png_compress_level,
                )
                write_placeholder_plot(
                    _plot_path("diag__ess_evolution_score"),
                    "Trace missing: ESS evolution unavailable.",
                    dpi=plot_dpi,
                    png_compress_level=png_compress_level,
                )
        if plots.pair_pwm or plots.parallel_pwm:
            if tf_pair is None and pairwise_placeholder_keys:
                if plots.pair_pwm:
                    write_placeholder_plot(
                        _plot_path("pwm__pair_scores"),
                        "Pairwise plot unavailable: tf_pair not resolved.",
                        dpi=plot_dpi,
                        png_compress_level=png_compress_level,
                    )
                if plots.parallel_pwm:
                    write_placeholder_plot(
                        _plot_path("pwm__parallel_scores"),
                        "Pairwise plot unavailable: tf_pair not resolved.",
                        dpi=plot_dpi,
                        png_compress_level=png_compress_level,
                    )
            else:
                idata_pair = make_pair_idata(sample_dir, tf_pair=tf_pair, sequences_df=seq_df)
                if plots.pair_pwm:
                    plot_pair_pwm_scores(
                        idata_pair,
                        plots_dir,
                        tf_pair=tf_pair,
                        dpi=plot_dpi,
                        png_compress_level=png_compress_level,
                        plot_format=plot_format,
                    )
                if plots.parallel_pwm:
                    plot_parallel_pwm_scores(
                        idata_pair,
                        plots_dir,
                        tf_pair=tf_pair,
                        dpi=plot_dpi,
                        png_compress_level=png_compress_level,
                        plot_format=plot_format,
                    )
        if plots.worst_tf_trace:
            plot_worst_tf_trace(
                seq_df,
                tf_names,
                _plot_path("plot__worst_tf_trace"),
                dpi=plot_dpi,
                png_compress_level=png_compress_level,
            )
        if plots.worst_tf_identity:
            plot_worst_tf_identity(
                seq_df,
                tf_names,
                _plot_path("plot__worst_tf_identity"),
                dpi=plot_dpi,
                png_compress_level=png_compress_level,
            )
        if plots.elite_filter_waterfall:
            plot_elite_filter_waterfall(
                elites_meta,
                _plot_path("plot__elite_filter_waterfall"),
                dpi=plot_dpi,
                png_compress_level=png_compress_level,
            )
        if plots.dashboard:
            plot_dashboard(
                seq_df,
                elites_df,
                tf_names,
                overlap_summary_df,
                elite_overlap_df,
                _plot_path("plot__dashboard"),
                dpi=plot_dpi,
                png_compress_level=png_compress_level,
            )

        # ── scatter plot ──────────────────────────────────────────────────────
        if plots.scatter_pwm:
            if tf_pair is None and pairwise_placeholder_keys:
                write_placeholder_plot(
                    _plot_path("pwm__scatter"),
                    "Scatter plot unavailable: tf_pair not resolved.",
                    dpi=plot_dpi,
                    png_compress_level=png_compress_level,
                )
            else:
                if per_pwm_path is None or not per_pwm_path.exists():
                    raise FileNotFoundError(f"Missing per-PWM score table in {tables_dir}")
                annotation = (
                    f"chains = {sample_meta.chains}\n"
                    f"iters  = {sample_meta.tune + sample_meta.draws}\n"
                    f"S/B/M   = {sample_meta.move_probs['S']:.2f}/"
                    f"{sample_meta.move_probs['B']:.2f}/"
                    f"{sample_meta.move_probs['M']:.2f}\n"
                    f"cooling = {sample_meta.cooling_kind}"
                )
                if any(sample_meta.move_probs.get(k, 0.0) > 0 for k in ("L", "W", "I")):
                    annotation = (
                        annotation
                        + "\n"
                        + f"L/W/I   = {sample_meta.move_probs.get('L', 0.0):.2f}/"
                        + f"{sample_meta.move_probs.get('W', 0.0):.2f}/"
                        + f"{sample_meta.move_probs.get('I', 0.0):.2f}"
                    )
                plot_scatter(
                    sample_dir,
                    pwms,
                    cfg_effective,
                    tf_pair=tf_pair,
                    per_pwm_path=per_pwm_path,
                    out_dir=plots_dir,
                    bidirectional=bidirectional,
                    pwm_sum_threshold=sample_meta.pwm_sum_threshold,
                    annotation=annotation,
                    output_format=plot_format,
                    elites_df=elites_df,
                    seq_len=int(manifest.get("sequence_length") or 0),
                    dpi=plot_dpi,
                    png_compress_level=png_compress_level,
                    pseudocounts=scoring_pseudocounts,
                    log_odds_clip=scoring_log_odds_clip,
                )

        if plots.score_hist:
            plot_score_hist(
                score_df,
                tf_names,
                _plot_path("score__hist"),
                dpi=plot_dpi,
                png_compress_level=png_compress_level,
            )
        if plots.score_box:
            plot_score_box(
                score_df,
                tf_names,
                _plot_path("score__box"),
                dpi=plot_dpi,
                png_compress_level=png_compress_level,
            )
        if plots.correlation_heatmap:
            plot_correlation_heatmap(
                score_df,
                tf_names,
                _plot_path("score__correlation"),
                dpi=plot_dpi,
                png_compress_level=png_compress_level,
            )
        if plots.pairgrid:
            plot_score_pairgrid(
                score_df,
                tf_names,
                _plot_path("score__pairgrid"),
                dpi=plot_dpi,
                png_compress_level=png_compress_level,
            )
        if plots.parallel_coords:
            if elites_df.empty:
                write_placeholder_plot(
                    _plot_path("elites__parallel_coords"),
                    "Parallel coordinates unavailable: no elites.",
                    dpi=plot_dpi,
                    png_compress_level=png_compress_level,
                )
            else:
                plot_parallel_coords(
                    elites_df,
                    tf_names,
                    _plot_path("elites__parallel_coords"),
                    dpi=plot_dpi,
                    png_compress_level=png_compress_level,
                )

        if plots.overlap_heatmap:
            if overlap_summary_df is None or overlap_summary_df.empty:
                write_placeholder_plot(
                    _plot_path("plot__overlap_heatmap"),
                    "Overlap heatmap unavailable: no elite overlap summary.",
                    dpi=plot_dpi,
                    png_compress_level=png_compress_level,
                )
            else:
                plot_overlap_heatmap(
                    overlap_summary_df,
                    tf_names,
                    _plot_path("plot__overlap_heatmap"),
                    dpi=plot_dpi,
                    png_compress_level=png_compress_level,
                )
        if plots.overlap_bp_distribution:
            if elite_overlap_df is None or elite_overlap_df.empty:
                write_placeholder_plot(
                    _plot_path("plot__overlap_bp_distribution"),
                    "Overlap distribution unavailable: no elite overlap rows.",
                    dpi=plot_dpi,
                    png_compress_level=png_compress_level,
                )
            else:
                plot_overlap_bp_distribution(
                    elite_overlap_df,
                    _plot_path("plot__overlap_bp_distribution"),
                    dpi=plot_dpi,
                    png_compress_level=png_compress_level,
                )
        if plots.overlap_strand_combos:
            if overlap_summary_df is None or overlap_summary_df.empty:
                write_placeholder_plot(
                    _plot_path("plot__overlap_strand_combos"),
                    "Overlap strand combos unavailable: no overlap summary.",
                    dpi=plot_dpi,
                    png_compress_level=png_compress_level,
                )
            else:
                plot_overlap_strand_combos(
                    overlap_summary_df,
                    _plot_path("plot__overlap_strand_combos"),
                    dpi=plot_dpi,
                    png_compress_level=png_compress_level,
                )
        if plots.motif_offset_rug:
            plot_motif_offset_rug(
                elites_df,
                tf_names,
                _plot_path("plot__motif_offset_rug"),
                pwm_widths=pwm_widths,
                dpi=plot_dpi,
                png_compress_level=png_compress_level,
            )
        if plots.pt_swap_by_pair and pt_swap_pairs_df is not None:
            plot_pt_swap_by_pair(
                pt_swap_pairs_df,
                _plot_path("plot__pt_swap_by_pair"),
                dpi=plot_dpi,
                png_compress_level=png_compress_level,
            )
        if plots.move_acceptance_time and move_stats_df is not None:
            plot_move_acceptance_time(
                move_stats_df,
                _plot_path("plot__move_acceptance_time"),
                dpi=plot_dpi,
                png_compress_level=png_compress_level,
            )
        if plots.move_usage_time and move_stats_df is not None:
            plot_move_usage_time(
                move_stats_df,
                _plot_path("plot__move_usage_time"),
                dpi=plot_dpi,
                png_compress_level=png_compress_level,
            )

        # ── top-5 tabular summary (unchanged logic – dynamic TF names) ────────
        if not elites_df.empty:
            if "rank" not in elites_df.columns:
                raise ValueError(
                    "Elites parquet missing 'rank' column. Re-run `cruncher sample` to regenerate elites.parquet."
                )
            if tf_pair is not None:
                x_tf, y_tf = tf_pair
                if f"score_{x_tf}" not in elites_df.columns or f"score_{y_tf}" not in elites_df.columns:
                    raise ValueError(
                        "Elites parquet missing score columns. Re-run `cruncher sample` to regenerate elites.parquet."
                    )
                logger.debug("Top-5 elites (%s & %s):", x_tf, y_tf)
                for _, row in elites_df.nsmallest(5, "rank").iterrows():
                    logger.debug(
                        "rank=%d seq=%s %s=%.1f %s=%.1f",
                        int(row["rank"]),
                        row["sequence"],
                        x_tf,
                        row[f"score_{x_tf}"],
                        y_tf,
                        row[f"score_{y_tf}"],
                    )
            elif len(tf_names) == 1:
                x_tf = tf_names[0]
                if f"score_{x_tf}" not in elites_df.columns:
                    raise ValueError(
                        "Elites parquet missing score columns. Re-run `cruncher sample` to regenerate elites.parquet."
                    )
                logger.debug("Top-5 elites (%s):", x_tf)
                for _, row in elites_df.nsmallest(5, "rank").iterrows():
                    logger.debug(
                        "rank=%d seq=%s %s=%.1f",
                        int(row["rank"]),
                        row["sequence"],
                        x_tf,
                        row[f"score_{x_tf}"],
                    )
            elif len(tf_names) > 1:
                missing_scores = [f"score_{tf}" for tf in tf_names if f"score_{tf}" not in elites_df.columns]
                if missing_scores:
                    raise ValueError(
                        "Elites parquet missing score columns. Re-run `cruncher sample` to regenerate elites.parquet."
                    )
                logger.debug("Top-5 elites (all TFs):")
                for _, row in elites_df.nsmallest(5, "rank").iterrows():
                    score_blob = " ".join(f"{tf}={row[f'score_{tf}']:.1f}" for tf in tf_names)
                    logger.debug(
                        "rank=%d seq=%s %s",
                        int(row["rank"]),
                        row["sequence"],
                        score_blob,
                    )
            else:
                logger.warning("No regulators configured; skipping elite summary.")

        plot_manifest = {
            "analysis_id": analysis_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "plots": [],
        }
        plot_artifacts: list[dict[str, object]] = []
        for spec in PLOT_SPECS:
            enabled_flag = getattr(plots, spec.key, False)
            spec_outputs = [out.replace("{ext}", plot_format) for out in spec.outputs]
            outputs = []
            missing = []
            for output in spec_outputs:
                out_path = analysis_root / output
                exists = out_path.exists()
                outputs.append({"path": output, "exists": exists})
                if enabled_flag and not exists:
                    missing.append(output)
                elif enabled_flag:
                    kind = "plot" if out_path.suffix.lower() in {".png", ".pdf", ".svg"} else "text"
                    plot_artifacts.append(
                        artifact_entry(
                            out_path,
                            sample_dir,
                            kind=kind,
                            label=f"{spec.label} ({out_path.name})",
                            stage="analysis",
                        )
                    )
            plot_manifest["plots"].append(
                {
                    "key": spec.key,
                    "label": spec.label,
                    "group": spec.group,
                    "description": spec.description,
                    "requires": list(spec.requires),
                    "enabled": enabled_flag,
                    "outputs": outputs,
                    "missing_outputs": missing,
                    "generated": enabled_flag and any(out["exists"] for out in outputs),
                }
            )
        if auto_opt_plot_path is not None:
            plot_manifest["plots"].append(
                {
                    "key": "auto_opt_tradeoffs",
                    "label": f"Auto-opt tradeoffs ({plot_format.upper()})",
                    "group": "auto_opt",
                    "description": "Balance vs top-K median score across auto-opt pilots.",
                    "requires": [],
                    "enabled": True,
                    "outputs": [
                        {
                            "path": str(auto_opt_plot_path.relative_to(sample_dir)),
                            "exists": auto_opt_plot_path.exists(),
                        }
                    ],
                    "missing_outputs": []
                    if auto_opt_plot_path.exists()
                    else [str(auto_opt_plot_path.relative_to(sample_dir))],
                    "generated": auto_opt_plot_path.exists(),
                }
            )
            plot_artifacts.append(
                artifact_entry(
                    auto_opt_plot_path,
                    sample_dir,
                    kind="plot",
                    label=f"Auto-opt tradeoffs ({plot_format.upper()})",
                    stage="analysis",
                )
            )

        plot_manifest_file = plot_manifest_path(analysis_root)
        plot_manifest_file.parent.mkdir(parents=True, exist_ok=True)
        plot_manifest_file.write_text(json.dumps(plot_manifest, indent=2))

        table_label_suffix = "Parquet" if table_ext == "parquet" else "CSV"
        tables_manifest_entries: list[dict[str, object]] = []
        if per_pwm_path is not None:
            tables_manifest_entries.append(
                {
                    "key": "per_pwm",
                    "label": f"Per-PWM scores ({table_label_suffix})",
                    "path": str(per_pwm_path.relative_to(sample_dir)),
                    "exists": per_pwm_path.exists(),
                }
            )
        tables_manifest_entries.extend(
            [
                {
                    "key": "score_summary",
                    "label": f"Per-TF summary ({table_label_suffix})",
                    "path": str(score_summary_path.relative_to(sample_dir)),
                    "exists": score_summary_path.exists(),
                },
                {
                    "key": "elite_topk",
                    "label": f"Elite top-K ({table_label_suffix})",
                    "path": str(topk_path.relative_to(sample_dir)),
                    "exists": topk_path.exists(),
                },
                {
                    "key": "joint_metrics",
                    "label": f"Joint score metrics ({table_label_suffix})",
                    "path": str(joint_metrics_path.relative_to(sample_dir)),
                    "exists": joint_metrics_path.exists(),
                },
                {
                    "key": "overlap_summary",
                    "label": f"Overlap summary ({table_label_suffix})",
                    "path": str(overlap_summary_path.relative_to(sample_dir)),
                    "exists": overlap_summary_path.exists(),
                },
                {
                    "key": "elite_overlap",
                    "label": f"Elite overlap details ({table_label_suffix})",
                    "path": str(elite_overlap_path.relative_to(sample_dir)),
                    "exists": elite_overlap_path.exists(),
                },
                {
                    "key": "diagnostics",
                    "label": "Diagnostics summary (JSON)",
                    "path": str(diagnostics_path.relative_to(sample_dir)),
                    "exists": diagnostics_path.exists(),
                },
                {
                    "key": "objective_components",
                    "label": "Objective components (JSON)",
                    "path": str(objective_components_path.relative_to(sample_dir)),
                    "exists": objective_components_path.exists(),
                },
            ]
        )
        table_manifest = {
            "analysis_id": analysis_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tables": tables_manifest_entries,
        }
        if move_stats_summary_path is not None:
            table_manifest["tables"].append(
                {
                    "key": "move_stats_summary",
                    "label": f"Move stats summary ({table_label_suffix})",
                    "path": str(move_stats_summary_path.relative_to(sample_dir)),
                    "exists": move_stats_summary_path.exists(),
                }
            )
        if move_stats_path is not None:
            table_manifest["tables"].append(
                {
                    "key": "move_stats",
                    "label": f"Move stats ({table_label_suffix})",
                    "path": str(move_stats_path.relative_to(sample_dir)),
                    "exists": move_stats_path.exists(),
                }
            )
        if pt_swap_pairs_path is not None:
            table_manifest["tables"].append(
                {
                    "key": "pt_swap_pairs",
                    "label": f"PT swap by pair ({table_label_suffix})",
                    "path": str(pt_swap_pairs_path.relative_to(sample_dir)),
                    "exists": pt_swap_pairs_path.exists(),
                }
            )
        if auto_opt_table_path is not None:
            table_manifest["tables"].append(
                {
                    "key": "auto_opt_pilots",
                    "label": f"Auto-opt pilot scorecard ({table_label_suffix})",
                    "path": str(auto_opt_table_path.relative_to(sample_dir)),
                    "exists": auto_opt_table_path.exists(),
                }
            )
        table_manifest_file = table_manifest_path(analysis_root)
        table_manifest_file.parent.mkdir(parents=True, exist_ok=True)
        table_manifest_file.write_text(json.dumps(table_manifest, indent=2))

        def _plot_reason(key: str) -> str:
            if key in tier0_plot_keys:
                return "default"
            if key in mcmc_plot_keys:
                return "mcmc_diagnostics"
            return "extra_plots"

        def _table_reason(key: str) -> str:
            if key in {"move_stats", "pt_swap_pairs"}:
                return "mcmc_diagnostics"
            if key in {"move_stats_summary"}:
                return "default"
            if key in {"auto_opt_pilots"}:
                return "extra_tables"
            if key in {"per_pwm"}:
                return "scatter_pwm"
            return "default"

        analysis_manifest_entries: list[dict[str, object]] = [
            {
                "path": str(analysis_used_file.relative_to(sample_dir)),
                "kind": "config",
                "label": "Analysis settings",
                "reason": "default",
                "exists": analysis_used_file.exists(),
            },
            {
                "path": str(plot_manifest_file.relative_to(sample_dir)),
                "kind": "manifest",
                "label": "Plot manifest",
                "reason": "default",
                "exists": plot_manifest_file.exists(),
            },
            {
                "path": str(table_manifest_file.relative_to(sample_dir)),
                "kind": "manifest",
                "label": "Table manifest",
                "reason": "default",
                "exists": table_manifest_file.exists(),
            },
        ]
        for table in table_manifest.get("tables", []):
            if not isinstance(table, dict):
                continue
            key = str(table.get("key") or "")
            path = table.get("path")
            analysis_manifest_entries.append(
                {
                    "path": path,
                    "kind": "table",
                    "label": table.get("label"),
                    "reason": _table_reason(key),
                    "exists": bool(table.get("exists")),
                    "key": key,
                }
            )
        for plot in plot_manifest.get("plots", []):
            if not isinstance(plot, dict):
                continue
            key = str(plot.get("key") or "")
            enabled = bool(plot.get("enabled"))
            for output in plot.get("outputs", []):
                if not isinstance(output, dict):
                    continue
                if not enabled and not output.get("exists"):
                    continue
                analysis_manifest_entries.append(
                    {
                        "path": output.get("path"),
                        "kind": "plot",
                        "label": plot.get("label"),
                        "reason": _plot_reason(key),
                        "exists": bool(output.get("exists")),
                        "key": key,
                    }
                )

        analysis_manifest_file = analysis_manifest_path(analysis_root)
        analysis_manifest_payload = {
            "analysis_id": analysis_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "extra_plots": analysis_cfg.extra_plots,
            "extra_tables": analysis_cfg.extra_tables,
            "mcmc_diagnostics": analysis_cfg.mcmc_diagnostics,
            "artifacts": analysis_manifest_entries,
        }
        analysis_manifest_file.write_text(json.dumps(analysis_manifest_payload, indent=2))

        artifacts: list[dict[str, object]] = [
            artifact_entry(
                analysis_used_file,
                sample_dir,
                kind="config",
                label="Analysis settings",
                stage="analysis",
            ),
        ]
        if per_pwm_path is not None:
            artifacts.append(
                artifact_entry(
                    per_pwm_path,
                    sample_dir,
                    kind="table",
                    label=f"Per-PWM scores ({table_label_suffix})",
                    stage="analysis",
                )
            )
        artifacts.extend(
            [
                artifact_entry(
                    score_summary_path,
                    sample_dir,
                    kind="table",
                    label=f"Per-TF summary ({table_label_suffix})",
                    stage="analysis",
                ),
                artifact_entry(
                    topk_path,
                    sample_dir,
                    kind="table",
                    label=f"Elite top-K ({table_label_suffix})",
                    stage="analysis",
                ),
                artifact_entry(
                    joint_metrics_path,
                    sample_dir,
                    kind="table",
                    label=f"Joint score metrics ({table_label_suffix})",
                    stage="analysis",
                ),
                artifact_entry(
                    overlap_summary_path,
                    sample_dir,
                    kind="table",
                    label=f"Overlap summary ({table_label_suffix})",
                    stage="analysis",
                ),
                artifact_entry(
                    elite_overlap_path,
                    sample_dir,
                    kind="table",
                    label=f"Elite overlap details ({table_label_suffix})",
                    stage="analysis",
                ),
                artifact_entry(
                    diagnostics_path,
                    sample_dir,
                    kind="json",
                    label="Diagnostics summary (JSON)",
                    stage="analysis",
                ),
                artifact_entry(
                    objective_components_path,
                    sample_dir,
                    kind="json",
                    label="Objective components (JSON)",
                    stage="analysis",
                ),
            ]
        )
        if move_stats_summary_path is not None:
            artifacts.append(
                artifact_entry(
                    move_stats_summary_path,
                    sample_dir,
                    kind="table",
                    label=f"Move stats summary ({table_label_suffix})",
                    stage="analysis",
                )
            )
        if move_stats_path is not None:
            artifacts.append(
                artifact_entry(
                    move_stats_path,
                    sample_dir,
                    kind="table",
                    label=f"Move stats ({table_label_suffix})",
                    stage="analysis",
                )
            )
        if pt_swap_pairs_path is not None:
            artifacts.append(
                artifact_entry(
                    pt_swap_pairs_path,
                    sample_dir,
                    kind="table",
                    label=f"PT swap by pair ({table_label_suffix})",
                    stage="analysis",
                )
            )
        if auto_opt_table_path is not None:
            artifacts.append(
                artifact_entry(
                    auto_opt_table_path,
                    sample_dir,
                    kind="table",
                    label=f"Auto-opt pilot scorecard ({table_label_suffix})",
                    stage="analysis",
                )
            )
        if auto_opt_plot_path is not None:
            artifacts.append(
                artifact_entry(
                    auto_opt_plot_path,
                    sample_dir,
                    kind="plot",
                    label="Auto-opt tradeoffs (PNG)",
                    stage="analysis",
                )
            )
        artifacts.extend(
            [
                artifact_entry(
                    plot_manifest_file,
                    sample_dir,
                    kind="json",
                    label="Plot manifest",
                    stage="analysis",
                ),
                artifact_entry(
                    table_manifest_file,
                    sample_dir,
                    kind="json",
                    label="Table manifest",
                    stage="analysis",
                ),
                artifact_entry(
                    analysis_manifest_file,
                    sample_dir,
                    kind="json",
                    label="Analysis manifest",
                    stage="analysis",
                ),
            ]
        )
        artifacts.extend(plot_artifacts)

        inputs_payload = {
            "sequences.parquet": {
                "path": str(seq_path.relative_to(sample_dir)),
                "sha256": sha256_path(seq_path),
            },
            "elites.parquet": {
                "path": str(elites_path.relative_to(sample_dir)),
                "sha256": sha256_path(elites_path),
            },
            "config_used.yaml": {
                "path": str(config_used_path(sample_dir).relative_to(sample_dir)),
                "sha256": sha256_path(config_used_path(sample_dir)),
            },
        }
        if trace_file.exists():
            inputs_payload["trace.nc"] = {
                "path": str(trace_file.relative_to(sample_dir)),
                "sha256": sha256_path(trace_file),
            }
        summary_payload = {
            "analysis_id": analysis_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run": run_name,
            "run_dir": str(sample_dir.resolve()),
            "analysis_dir": str(analysis_root.resolve()),
            "tf_names": tf_names,
            "diagnostics": diagnostics_summary,
            "analysis_layout_version": ANALYSIS_LAYOUT_VERSION,
            "analysis_config": analysis_cfg.model_dump(),
            "cruncher_version": _get_version(),
            "git_commit": _get_git_commit(config_path),
            "analysis_used": str(analysis_used_file.relative_to(sample_dir)),
            "config_used": str(config_used_path(sample_dir).relative_to(sample_dir)),
            "plot_manifest": str(plot_manifest_file.relative_to(sample_dir)),
            "table_manifest": str(table_manifest_file.relative_to(sample_dir)),
            "analysis_manifest": str(analysis_manifest_file.relative_to(sample_dir)),
            "inputs": inputs_payload,
            "signature": analysis_signature,
            "signature_inputs": signature_payload,
            "artifacts": [item["path"] for item in artifacts],
            "objective_components": objective_components,
            "overlap_summary": overlap_summary,
        }
        summary_file = summary_path(analysis_root)
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        report_json, report_md = ensure_report(
            analysis_root=analysis_root,
            summary_payload=summary_payload,
            diagnostics_payload=diagnostics_summary,
            objective_components=objective_components,
            overlap_summary=overlap_summary,
            analysis_used_payload=analysis_used_payload,
            refresh=True,
        )
        summary_payload["report_json"] = str(report_json.relative_to(sample_dir))
        summary_payload["report_md"] = str(report_md.relative_to(sample_dir))
        summary_file.write_text(json.dumps(summary_payload, indent=2))
        analysis_manifest_entries.append(
            {
                "path": str(summary_file.relative_to(sample_dir)),
                "kind": "summary",
                "label": "Analysis summary",
                "reason": "default",
                "exists": summary_file.exists(),
            }
        )
        analysis_manifest_entries.extend(
            [
                {
                    "path": str(report_json.relative_to(sample_dir)),
                    "kind": "report",
                    "label": "Analysis report (JSON)",
                    "reason": "default",
                    "exists": report_json.exists(),
                    "key": "report_json",
                },
                {
                    "path": str(report_md.relative_to(sample_dir)),
                    "kind": "report",
                    "label": "Analysis report (Markdown)",
                    "reason": "default",
                    "exists": report_md.exists(),
                    "key": "report_md",
                },
            ]
        )
        analysis_manifest_payload["artifacts"] = analysis_manifest_entries
        analysis_manifest_file.write_text(json.dumps(analysis_manifest_payload, indent=2))
        artifacts.append(
            artifact_entry(
                summary_file,
                sample_dir,
                kind="json",
                label="Analysis summary",
                stage="analysis",
            )
        )
        artifacts.extend(
            [
                artifact_entry(
                    report_json,
                    sample_dir,
                    kind="json",
                    label="Analysis report (JSON)",
                    stage="analysis",
                ),
                artifact_entry(
                    report_md,
                    sample_dir,
                    kind="text",
                    label="Analysis report (Markdown)",
                    stage="analysis",
                ),
            ]
        )

        append_artifacts(manifest, artifacts)
        analysis_root.mkdir(parents=True, exist_ok=True)

        from dnadesign.cruncher.app.run_service import (
            update_run_index_from_manifest,
        )
        from dnadesign.cruncher.artifacts.manifest import write_manifest

        write_manifest(sample_dir, manifest)
        update_run_index_from_manifest(
            config_path,
            sample_dir,
            manifest,
            catalog_root=cfg.motif_store.catalog_root,
        )
        logger.info("Analysis artifacts recorded (%s).", analysis_id)
        analysis_runs.append(analysis_root)
    return analysis_runs
