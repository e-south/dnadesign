"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/__init__.py

Public DenseGen package API.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .analysis_surface import (
    ArtifactRecord,
    DenseGenAnalysisSurface,
    Diagnostic,
    FreshnessSummary,
    NotebookSurface,
    PlotTaxonomyEntry,
    RuntimeSummary,
    inspect_analysis_surface,
)
from .src.cli.main import app
from .src.cli.notebook_template_cells import build_run_summary_tables
from .src.config import ConfigError, LoadedConfig, load_config, resolve_outputs_scoped_path, resolve_run_root
from .src.integrations.baserender.notebook_contract import densegen_notebook_render_contract
from .src.viz.plot_registry import PLOT_SPECS

__all__ = [
    "ArtifactRecord",
    "ConfigError",
    "DenseGenAnalysisSurface",
    "Diagnostic",
    "FreshnessSummary",
    "LoadedConfig",
    "NotebookSurface",
    "PLOT_SPECS",
    "PlotTaxonomyEntry",
    "RuntimeSummary",
    "app",
    "build_run_summary_tables",
    "densegen_notebook_render_contract",
    "inspect_analysis_surface",
    "load_config",
    "resolve_outputs_scoped_path",
    "resolve_run_root",
]
