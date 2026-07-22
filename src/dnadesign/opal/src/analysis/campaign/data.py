"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/campaign/data.py

Campaign data containers for analysis consumers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...config.types import RootConfig
from ...storage.data_access import RecordsStore
from ...storage.workspace import CampaignWorkspace


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
