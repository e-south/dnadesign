"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/synthesis_handoff/campaigns.py

Stress OPAL campaign identity helpers for synthesis handoffs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_STRESS_OPAL_CAMPAIGN_CONFIG = Path("src/dnadesign/opal/campaigns/secg_msrb_greedy/configs/campaign.yaml")

STRESS_SELECTION_VIEW_ALIAS_CODES: dict[str, str] = {
    "ethanol": "ETH",
    "ciprofloxacin": "CIP",
    "and": "AND",
}
SFXI_SOURCE_CAMPAIGN_SELECTION_VIEWS: dict[str, str] = {
    "secg_ethanol_rf_sfxi_topn": "ethanol",
    "secg_cipro_rf_sfxi_topn": "ciprofloxacin",
    "secg_and_rf_sfxi_topn": "and",
}
STRESS_OPAL_SYNTHESIS_ALIAS_PREFIX = "SECG"


def stress_selection_view_code(selection_view_id: str) -> str:
    """Return the short synthesis alias code for a stress selection view."""

    view_id = str(selection_view_id).strip()
    code = STRESS_SELECTION_VIEW_ALIAS_CODES.get(view_id)
    if code is None:
        raise ValueError(f"unknown stress selection view for synthesis alias: {view_id}")
    return code


def sfxi_source_selection_view_id(campaign_slug: str) -> str:
    """Map a digest-pinned SFXI source campaign to its declared selection view."""

    slug = str(campaign_slug).strip()
    view_id = SFXI_SOURCE_CAMPAIGN_SELECTION_VIEWS.get(slug)
    if view_id is None:
        raise ValueError(f"unknown SFXI source campaign slug: {slug}")
    return view_id


def batch0_synthesis_name(campaign_slug: str, selection_rank: int) -> str:
    """Return the deterministic batch-0 synthesis alias for a campaign row."""

    if int(selection_rank) <= 0:
        raise ValueError("selection_rank must be positive")
    view_id = sfxi_source_selection_view_id(campaign_slug)
    return f"{STRESS_OPAL_SYNTHESIS_ALIAS_PREFIX}-B0-{stress_selection_view_code(view_id)}-{int(selection_rank):02d}"
