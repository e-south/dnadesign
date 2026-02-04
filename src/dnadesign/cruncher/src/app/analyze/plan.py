"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/analyze/plan.py

Resolve analysis plot settings and override policies.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from dnadesign.cruncher.analysis.plot_registry import PLOT_SPECS
from dnadesign.cruncher.config.schema_v2 import AnalysisConfig, CruncherConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisPlan:
    analysis_cfg: AnalysisConfig
    cfg_effective: CruncherConfig
    plot_dpi: int
    png_compress_level: int
    plot_format: str
    tier0_plot_keys: set[str]
    mcmc_plot_keys: set[str]
    override_payload: dict[str, object] | None
    auto_adjustments: dict[str, object]


def resolve_analysis_plan(
    cfg: CruncherConfig,
    *,
    plot_keys_override: list[str] | None = None,
    scatter_background_override: bool | None = None,
    scatter_background_samples_override: int | None = None,
    scatter_background_seed_override: int | None = None,
) -> AnalysisPlan:
    analysis_cfg = cfg.analysis
    if analysis_cfg is None:
        raise ValueError("analysis section is required for analyze")

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

    return AnalysisPlan(
        analysis_cfg=analysis_cfg,
        cfg_effective=cfg_effective,
        plot_dpi=plot_dpi,
        png_compress_level=png_compress_level,
        plot_format=plot_format,
        tier0_plot_keys=tier0_plot_keys,
        mcmc_plot_keys=mcmc_plot_keys,
        override_payload=override_payload,
        auto_adjustments=auto_adjustments,
    )
