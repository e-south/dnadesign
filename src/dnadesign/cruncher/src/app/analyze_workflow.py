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
    report_json_path,
    report_md_path,
    summary_path,
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
from dnadesign.cruncher.app.analyze.manifests import build_analysis_manifests
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
from dnadesign.cruncher.app.analyze.plan import resolve_analysis_plan
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
from dnadesign.cruncher.core.sequence import canon_string
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

    plan = resolve_analysis_plan(
        cfg,
        plot_keys_override=plot_keys_override,
        scatter_background_override=scatter_background_override,
        scatter_background_samples_override=scatter_background_samples_override,
        scatter_background_seed_override=scatter_background_seed_override,
    )
    analysis_cfg = plan.analysis_cfg
    cfg_effective = plan.cfg_effective
    plot_dpi = plan.plot_dpi
    png_compress_level = plan.png_compress_level
    plot_format = plan.plot_format
    tier0_plot_keys = plan.tier0_plot_keys
    mcmc_plot_keys = plan.mcmc_plot_keys
    override_payload = plan.override_payload
    auto_adjustments = plan.auto_adjustments

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
            raise FileNotFoundError(
                f"Missing artifacts/trace.nc in {sample_dir}. "
                "Re-run `cruncher sample` with sample.output.trace.save=true."
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
        analysis_created_at = datetime.now(timezone.utc).isoformat()
        analysis_used_payload = {
            "analysis_id": analysis_id,
            "created_at": analysis_created_at,
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
        }
        pvalue_cache = manifest.get("pvalue_cache")
        if isinstance(pvalue_cache, dict):
            sample_meta_payload["pvalue_cache"] = pvalue_cache
        if isinstance(manifest.get("early_stop"), dict):
            sample_meta_payload["early_stop"] = manifest.get("early_stop")
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
            early_stop=manifest.get("early_stop"),
        )
        objective_components_path.write_text(json.dumps(objective_components, indent=2))

        elites_mmr_summary_path = None
        mmr_summary_payload = elites_meta.get("mmr_summary") if isinstance(elites_meta, dict) else None
        if isinstance(mmr_summary_payload, dict) and mmr_summary_payload:

            def _canonical_unique_fraction(frame: pd.DataFrame) -> float | None:
                if frame is None or frame.empty or "sequence" not in frame.columns:
                    return None
                total = int(len(frame))
                if total == 0:
                    return None
                if "canonical_sequence" in frame.columns:
                    unique = int(frame["canonical_sequence"].astype(str).nunique())
                elif sample_meta.dsdna_canonicalize:
                    unique = int(frame["sequence"].astype(str).map(canon_string).nunique())
                else:
                    unique = int(frame["sequence"].astype(str).nunique())
                return unique / float(total)

            draw_unique_fraction = objective_components.get("unique_fraction_canonical") or objective_components.get(
                "unique_fraction_raw"
            )
            elite_unique_fraction = _canonical_unique_fraction(elites_df)
            mmr_summary_row = {
                "k": mmr_summary_payload.get("k"),
                "alpha": mmr_summary_payload.get("alpha"),
                "pool_size": mmr_summary_payload.get("pool_size"),
                "median_relevance_raw": mmr_summary_payload.get("median_relevance_raw"),
                "mean_pairwise_distance": mmr_summary_payload.get("mean_pairwise_distance"),
                "min_pairwise_distance": mmr_summary_payload.get("min_pairwise_distance"),
                "unique_fraction_canonical_draw": draw_unique_fraction,
                "unique_fraction_canonical_elites": elite_unique_fraction,
            }
            elites_mmr_summary_path = tables_dir / f"elites_mmr_summary.{table_ext}"
            mmr_summary_df = pd.DataFrame([mmr_summary_row])
            if table_ext == "parquet":
                write_parquet(mmr_summary_df, elites_mmr_summary_path)
            else:
                mmr_summary_df.to_csv(elites_mmr_summary_path, index=False)

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
                    score_metric = None
                    auto_cfg_payload = auto_opt_payload.get("auto_opt_config")
                    if isinstance(auto_cfg_payload, dict):
                        score_metric = auto_cfg_payload.get("scorecard_metric")
                    if score_metric == "elites_mmr":
                        score_col = "pilot_score"
                        score_label = "Pilot score (MMR)"
                    else:
                        score_col = "top_k_median_final" if "top_k_median_final" in df_auto.columns else "best_score"
                        score_label = "Top-K median score (final beta)"
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
                            ax.set_ylabel(score_label)
                            ax.set_title("Auto-opt tradeoffs")
                            if colors is not None:
                                fig.colorbar(scatter, ax=ax, label="Length")
                            fig.tight_layout()
                            savefig(fig, auto_opt_plot_path, dpi=plot_dpi, png_compress_level=png_compress_level)
                            plt.close(fig)
                    elif score_metric == "elites_mmr":
                        logger.warning("Auto-opt tradeoff plot skipped: pilot_score missing from scorecard.")

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

        manifest_bundle = build_analysis_manifests(
            analysis_id=analysis_id,
            created_at=analysis_created_at,
            analysis_root=analysis_root,
            sample_dir=sample_dir,
            analysis_used_file=analysis_used_file,
            plot_format=plot_format,
            plots=plots,
            tier0_plot_keys=tier0_plot_keys,
            mcmc_plot_keys=mcmc_plot_keys,
            extra_plots=analysis_cfg.extra_plots,
            extra_tables=analysis_cfg.extra_tables,
            mcmc_diagnostics=analysis_cfg.mcmc_diagnostics,
            per_pwm_path=per_pwm_path,
            score_summary_path=score_summary_path,
            topk_path=topk_path,
            joint_metrics_path=joint_metrics_path,
            overlap_summary_path=overlap_summary_path,
            elite_overlap_path=elite_overlap_path,
            diagnostics_path=diagnostics_path,
            objective_components_path=objective_components_path,
            elites_mmr_summary_path=elites_mmr_summary_path,
            move_stats_summary_path=move_stats_summary_path,
            move_stats_path=move_stats_path,
            pt_swap_pairs_path=pt_swap_pairs_path,
            auto_opt_table_path=auto_opt_table_path,
            auto_opt_plot_path=auto_opt_plot_path,
            table_ext=table_ext,
        )
        plot_manifest_file = manifest_bundle.plot_manifest_file
        table_manifest_file = manifest_bundle.table_manifest_file
        analysis_manifest_file = manifest_bundle.analysis_manifest_file
        analysis_manifest_payload = manifest_bundle.analysis_manifest_payload
        analysis_manifest_entries = manifest_bundle.analysis_manifest_entries
        artifacts = manifest_bundle.analysis_artifacts

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
