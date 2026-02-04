"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/core/reporting.py

Report data and rendering facade for DenseGen runs.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .reporting_data import ReportBundle, collect_report_data
from .reporting_render import write_report

__all__ = ["ReportBundle", "collect_report_data", "write_report"]
