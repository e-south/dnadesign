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
from .renderers.explain import render_explain_text
from .renderers.ingest import render_ingest_commit_text, render_ingest_preview_text, render_ingest_runtime_text
from .renderers.init import render_init_text
from .renderers.log import render_round_log_summary_text
from .renderers.model import render_model_show_text
from .renderers.record import render_record_report_text
from .renderers.run import render_run_meta_text, render_run_summary_text
from .renderers.runs import render_runs_list_text
from .renderers.status import render_status_text

__all__ = [
    "bullet_list",
    "kv_block",
    "short_array",
    "render_explain_text",
    "render_ingest_commit_text",
    "render_ingest_preview_text",
    "render_ingest_runtime_text",
    "render_init_text",
    "render_model_show_text",
    "render_record_report_text",
    "render_run_summary_text",
    "render_run_meta_text",
    "render_runs_list_text",
    "render_status_text",
    "render_round_log_summary_text",
]
