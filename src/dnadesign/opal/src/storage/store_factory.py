"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/storage/store_factory.py

Shared RecordsStore construction from a validated RootConfig.

Module Author(s): Eric J. South (extended by Codex)
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from ..config import LocationLocal, LocationUSR, RootConfig
from ..core.utils import OpalError
from .data_access import RecordsStore


def records_store_from_config(cfg: RootConfig) -> RecordsStore:
    """
    Build a RecordsStore from RootConfig without CLI dependencies.
    """
    loc = cfg.data.location
    if isinstance(loc, LocationUSR):
        records = Path(loc.path) / loc.dataset / "records.parquet"
        kind = "usr"
    elif isinstance(loc, LocationLocal):
        records = Path(loc.path)
        kind = "local"
    else:
        raise OpalError("Unknown data location kind.")

    return RecordsStore(
        kind=kind,
        records_path=records,
        campaign_slug=cfg.campaign.slug,
        x_col=cfg.data.x_column_name,
        y_col=cfg.data.y_column_name,
        x_transform_name=cfg.data.transforms_x.name,
        x_transform_params=cfg.data.transforms_x.params,
    )
