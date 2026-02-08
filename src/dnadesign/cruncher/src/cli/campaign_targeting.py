"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/campaign_targeting.py

Resolve campaign-driven runtime targeting for CLI commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dnadesign.cruncher.app.campaign_service import CampaignExpansion, expand_campaign
from dnadesign.cruncher.config.schema_v3 import CruncherConfig


@dataclass(frozen=True)
class RuntimeTargeting:
    cfg: CruncherConfig
    campaign: CampaignExpansion | None


def _config_with_regulator_sets(
    cfg: CruncherConfig,
    *,
    regulator_sets: list[list[str]],
    campaign: CampaignExpansion | None,
) -> CruncherConfig:
    payload = cfg.model_dump(mode="python")
    workspace_payload = payload.get("workspace")
    if not isinstance(workspace_payload, dict):
        raise ValueError("Config is missing workspace settings.")
    workspace_payload["regulator_sets"] = regulator_sets
    payload["workspace"] = workspace_payload
    if campaign is not None:
        payload["campaign"] = {
            "name": campaign.name,
            "campaign_id": campaign.campaign_id,
        }
    return CruncherConfig.model_validate(payload)


def resolve_runtime_targeting(
    *,
    cfg: CruncherConfig,
    config_path: Path,
    command_name: str,
    campaign_name: str | None,
) -> RuntimeTargeting:
    if campaign_name is not None:
        expansion = expand_campaign(
            cfg=cfg,
            config_path=config_path,
            campaign_name=campaign_name,
            include_metrics=False,
        )
        resolved_cfg = _config_with_regulator_sets(
            cfg,
            regulator_sets=expansion.regulator_sets,
            campaign=expansion,
        )
        return RuntimeTargeting(cfg=resolved_cfg, campaign=expansion)

    if cfg.regulator_sets:
        return RuntimeTargeting(cfg=cfg, campaign=None)

    if cfg.campaigns:
        raise ValueError(f"{command_name} requires --campaign when workspace.regulator_sets is empty.")
    raise ValueError(f"{command_name} requires at least one workspace.regulator_set or a configured campaign.")
