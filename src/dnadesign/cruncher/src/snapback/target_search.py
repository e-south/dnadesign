"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/snapback/target_search.py

Target-first snapback catalog search for exact preserved-site geometry hits.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.snapback.preserved_search.reporting import (
    render_target_search_markdown_report,
)
from dnadesign.cruncher.snapback.preserved_search.runner import (
    search_snapback_target_hits,
)

__all__ = [
    "render_target_search_markdown_report",
    "search_snapback_target_hits",
]
