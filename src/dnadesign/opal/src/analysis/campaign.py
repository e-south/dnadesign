"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/campaign.py

Loads campaign configuration and workspace paths for analysis. Provides IO-only
data containers for analysis consumers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from ..config.loader import load_config
from ..config.types import RootConfig
from ..core.config_resolve import resolve_campaign_config_path
from ..core.utils import OpalError
from ..storage.data_access import RecordsStore
from ..storage.store_factory import records_store_from_config
from ..storage.workspace import CampaignWorkspace


@dataclass(frozen=True)
class CampaignPaths:
    config_path: Path
    campaign_dir: Path
    outputs_dir: Path
    ledger_predictions_dir: Path
    ledger_runs_path: Path
    ledger_labels_path: Path

    @classmethod
    def from_workspace(cls, config_path: Path, ws: CampaignWorkspace) -> "CampaignPaths":
        return cls(
            config_path=config_path,
            campaign_dir=ws.workdir,
            outputs_dir=ws.outputs_dir,
            ledger_predictions_dir=ws.ledger_predictions_dir,
            ledger_runs_path=ws.ledger_runs_path,
            ledger_labels_path=ws.ledger_labels_path,
        )


@dataclass(frozen=True)
class CampaignData:
    config: RootConfig
    config_path: Path
    config_dict: dict
    workspace: CampaignWorkspace
    paths: CampaignPaths
    store: RecordsStore


def _load_config_dict(path: Path) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise OpalError(f"Campaign YAML did not parse to a mapping: {path}")
    return cfg


def load_campaign_data(config_opt: Optional[Path], *, allow_dir: bool = False) -> CampaignData:
    cfg_path = resolve_campaign_config_path(config_opt, allow_dir=allow_dir)
    cfg = load_config(cfg_path)
    ws = CampaignWorkspace.from_config(cfg, cfg_path)
    cfg_dict = _load_config_dict(cfg_path)
    store = records_store_from_config(cfg)
    paths = CampaignPaths.from_workspace(cfg_path, ws)
    return CampaignData(
        config=cfg,
        config_path=cfg_path,
        config_dict=cfg_dict,
        workspace=ws,
        paths=paths,
        store=store,
    )
