"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/analyze/plan.py

Resolve the analysis plan for the curated v3 plot/table suite.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

from dnadesign.cruncher.config.schema_v3 import AnalysisConfig, CruncherConfig


@dataclass(frozen=True)
class AnalysisPlan:
    analysis_cfg: AnalysisConfig
    cfg_effective: CruncherConfig
    plot_dpi: int
    plot_format: str
    table_format: str
    override_payload: dict[str, object] | None


def resolve_analysis_plan(cfg: CruncherConfig) -> AnalysisPlan:
    analysis_cfg = cfg.analysis if cfg.analysis is not None else AnalysisConfig()
    return AnalysisPlan(
        analysis_cfg=analysis_cfg,
        cfg_effective=cfg,
        plot_dpi=int(analysis_cfg.plot_dpi),
        plot_format=str(analysis_cfg.plot_format),
        table_format=str(analysis_cfg.table_format),
        override_payload=None,
    )
