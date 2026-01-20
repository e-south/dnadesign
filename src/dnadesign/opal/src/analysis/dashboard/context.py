"""Shared dashboard context container."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .datasets import CampaignInfo


@dataclass(frozen=True)
class DashboardContext:
    dataset_path: Path | None
    dataset_name: str | None
    campaign_info: CampaignInfo | None
    workdir: Path | None
    ledger_runs_df: pl.DataFrame | None
    ledger_labels_df: pl.DataFrame | None
    ledger_preds_df: pl.DataFrame | None
