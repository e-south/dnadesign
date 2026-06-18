"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/campaign/__init__.py

Campaign analysis ontology: config, workspace, records, and campaign readers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .analysis import CampaignAnalysis
from .data import CampaignData, CampaignPaths
from .loading import load_campaign_data

__all__ = [
    "CampaignAnalysis",
    "CampaignData",
    "CampaignPaths",
    "load_campaign_data",
]
