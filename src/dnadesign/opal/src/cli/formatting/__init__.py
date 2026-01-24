"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/formatting/__init__.py

Exposes formatting helpers and renderers for OPAL CLI commands. Aggregates core
formatting utilities and per-command renderers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .core import bullet_list, kv_block, short_array
from .renderers.explain import render_explain_human
from .renderers.ingest import render_ingest_commit_human, render_ingest_preview_human
from .renderers.init import render_init_human
from .renderers.log import render_round_log_summary_human
from .renderers.model import render_model_show_human
from .renderers.record import render_record_report_human
from .renderers.run import render_run_meta_human, render_run_summary_human
from .renderers.runs import render_runs_list_human
from .renderers.status import render_status_human

__all__ = [
    "bullet_list",
    "kv_block",
    "short_array",
    "render_explain_human",
    "render_ingest_commit_human",
    "render_ingest_preview_human",
    "render_init_human",
    "render_model_show_human",
    "render_record_report_human",
    "render_run_summary_human",
    "render_run_meta_human",
    "render_runs_list_human",
    "render_status_human",
    "render_round_log_summary_human",
]
