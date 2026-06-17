"""Stress OPAL campaign identity helpers for synthesis handoffs."""

from __future__ import annotations

from pathlib import Path

DEFAULT_STRESS_OPAL_CAMPAIGN_CONFIGS: tuple[Path, ...] = (
    Path("src/dnadesign/opal/campaigns/stress_eth_cip_ethanol_rf_sfxi_topn/configs/campaign.yaml"),
    Path("src/dnadesign/opal/campaigns/stress_eth_cip_cipro_rf_sfxi_topn/configs/campaign.yaml"),
    Path("src/dnadesign/opal/campaigns/stress_eth_cip_and_rf_sfxi_topn/configs/campaign.yaml"),
)

STRESS_OPAL_CAMPAIGN_ALIAS_CODES: dict[str, str] = {
    "stress_eth_cip_ethanol_rf_sfxi_topn": "ETH",
    "stress_eth_cip_cipro_rf_sfxi_topn": "CIP",
    "stress_eth_cip_and_rf_sfxi_topn": "AND",
}
STRESS_OPAL_SYNTHESIS_ALIAS_PREFIX = "SECG"


def stress_opal_campaign_code(campaign_slug: str) -> str:
    """Return the short synthesis alias code for a known stress OPAL campaign."""

    slug = str(campaign_slug).strip()
    code = STRESS_OPAL_CAMPAIGN_ALIAS_CODES.get(slug)
    if code is None:
        raise ValueError(f"unknown stress OPAL campaign slug for synthesis alias: {slug}")
    return code


def batch0_synthesis_name(campaign_slug: str, selection_rank: int) -> str:
    """Return the deterministic batch-0 synthesis alias for a campaign row."""

    if int(selection_rank) <= 0:
        raise ValueError("selection_rank must be positive")
    return (
        f"{STRESS_OPAL_SYNTHESIS_ALIAS_PREFIX}-B0-{stress_opal_campaign_code(campaign_slug)}-{int(selection_rank):02d}"
    )


def opal_round_synthesis_name(campaign_slug: str, as_of_round: int, selection_rank: int) -> str:
    """Return the deterministic measured-round synthesis alias for a campaign row."""

    if int(as_of_round) < 0:
        raise ValueError("as_of_round must be non-negative")
    if int(selection_rank) <= 0:
        raise ValueError("selection_rank must be positive")
    return (
        f"{STRESS_OPAL_SYNTHESIS_ALIAS_PREFIX}-R{int(as_of_round)}-"
        f"{stress_opal_campaign_code(campaign_slug)}-{int(selection_rank):02d}"
    )
