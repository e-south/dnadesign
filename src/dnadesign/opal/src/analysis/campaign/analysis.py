"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/campaign/analysis.py

Campaign-scoped analysis reader object.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl

from ...config.types import RootConfig
from ...storage.workspace import CampaignWorkspace
from ..ledger import (
    RoundSelector,
    load_predictions_with_setpoint,
    read_labels,
    read_predictions,
    read_runs,
    read_selection_view_predictions,
    scan_labels,
    scan_predictions,
    scan_runs,
)
from .data import CampaignData, CampaignPaths
from .loading import load_campaign_data


@dataclass(frozen=True)
class CampaignAnalysis:
    data: CampaignData

    @classmethod
    def from_config_path(cls, config_opt: Path | None, *, allow_dir: bool = False) -> "CampaignAnalysis":
        data = load_campaign_data(config_opt, allow_dir=allow_dir)
        return cls(data=data)

    @property
    def config_path(self) -> Path:
        return self.data.config_path

    @property
    def config(self) -> RootConfig:
        return self.data.config

    @property
    def workspace(self) -> CampaignWorkspace:
        return self.data.workspace

    @property
    def paths(self) -> CampaignPaths:
        return self.data.paths

    def records_store(self):
        return self.data.store

    def read_config_dict(self) -> dict:
        return dict(self.data.config_dict)

    def read_runs(self) -> pl.DataFrame:
        return read_runs(self.workspace.ledger_runs_path)

    def read_labels(self) -> pl.DataFrame:
        return read_labels(self.workspace.ledger_labels_path)

    def scan_runs(self) -> pl.LazyFrame:
        return scan_runs(self.workspace.ledger_runs_path)

    def scan_labels(self) -> pl.LazyFrame:
        return scan_labels(self.workspace.ledger_labels_path)

    def scan_predictions(self) -> pl.LazyFrame:
        return scan_predictions(self.workspace.ledger_predictions_dir)

    def read_predictions(
        self,
        *,
        columns: Sequence[str] | None = None,
        round_selector: RoundSelector | None = None,
        run_id: str | None = None,
        runs_df: pl.DataFrame | None = None,
        allow_missing: bool = False,
        require_run_id: bool = True,
    ) -> pl.DataFrame:
        if runs_df is None:
            runs_df = self.read_runs()
        return read_predictions(
            self.workspace.ledger_predictions_dir,
            columns=columns,
            round_selector=round_selector,
            run_id=run_id,
            runs_df=runs_df,
            allow_missing=allow_missing,
            require_run_id=require_run_id,
        )

    def predictions_with_setpoint(
        self,
        columns: Iterable[str],
        *,
        selection_view_id: str,
        round_selector: RoundSelector | None = None,
        run_id: str | None = None,
        require_run_id: bool = True,
    ) -> pl.DataFrame:
        return load_predictions_with_setpoint(
            self.workspace.outputs_dir,
            columns,
            selection_view_id=selection_view_id,
            round_selector=round_selector,
            run_id=run_id,
            require_run_id=require_run_id,
        )

    def read_selection_view_predictions(
        self,
        *,
        selection_view_id: str,
        columns: Sequence[str] | None = None,
        round_selector: RoundSelector | None = None,
        run_id: str | None = None,
        runs_df: pl.DataFrame | None = None,
        require_run_id: bool = True,
    ) -> pl.DataFrame:
        if runs_df is None:
            runs_df = self.read_runs()
        return read_selection_view_predictions(
            self.workspace.ledger_predictions_dir,
            selection_view_id=selection_view_id,
            columns=columns,
            round_selector=round_selector,
            run_id=run_id,
            runs_df=runs_df,
            require_run_id=require_run_id,
        )
