"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/__init__.py

Public OPAL package API.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .src.config.loader import load_config
from .src.reporting.predictions import read_campaign_predictions
from .src.reporting.progress import build_campaign_progress, render_campaign_progress_text
from .src.reporting.review import build_campaign_review
from .src.storage.x_contracts import validate_x_parquet_column

__all__ = [
    "build_campaign_progress",
    "build_campaign_review",
    "load_config",
    "read_campaign_predictions",
    "render_campaign_progress_text",
    "validate_x_parquet_column",
]
