"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/campaign/loading.py

Campaign configuration and workspace loading for analysis.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ...config.loader import load_config
from ...core.config_resolve import resolve_campaign_config_path
from ...core.utils import OpalError
from ...storage.store_factory import records_store_from_config
from ...storage.workspace import CampaignWorkspace
from .data import CampaignData, CampaignPaths


def load_campaign_data(
    config_opt: Path | None,
    *,
    allow_dir: bool = False,
    usr_root: Path | str | None = None,
) -> CampaignData:
    cfg_path = resolve_campaign_config_path(config_opt, allow_dir=allow_dir)
    cfg = load_config(cfg_path, usr_root=usr_root)
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


def _load_config_dict(path: Path) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise OpalError(f"Campaign YAML did not parse to a mapping: {path}")
    return cfg
