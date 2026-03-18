"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/execution.py

Thin cluster execution facade that re-exports shared command runtimes.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .analysis.contracts import AnalysisRequest
from .execution_analysis import run_analyze
from .execution_fit import run_fit
from .execution_support import (
    CommandExecution,
    append_command_record_or_warn,
    assert_preserve_columns,
    attach_columns_schema_preserving,
    cluster_overlay_col,
    context_and_df,
    intra_sim_overlay_col,
    load_highlight_ids_from_file,
    print_fit_summary,
    progress_scope,
    resolve_color_by,
    resolve_scoped_out_dir,
)
from .execution_sweep import run_sweep
from .execution_table import run_delete_columns, run_intra_similarity
from .execution_umap import run_umap
from .presets.runtime import apply_plot_preset, apply_preset

__all__ = [
    "CommandExecution",
    "AnalysisRequest",
    "append_command_record_or_warn",
    "apply_plot_preset",
    "apply_preset",
    "assert_preserve_columns",
    "attach_columns_schema_preserving",
    "cluster_overlay_col",
    "context_and_df",
    "intra_sim_overlay_col",
    "load_highlight_ids_from_file",
    "print_fit_summary",
    "progress_scope",
    "resolve_color_by",
    "resolve_scoped_out_dir",
    "run_analyze",
    "run_delete_columns",
    "run_fit",
    "run_intra_similarity",
    "run_sweep",
    "run_umap",
]
