"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/plots/_context.py

PlotContext — context object passed to plot plugins.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..workspace import CampaignWorkspace


@dataclass
class PlotContext:
    campaign_dir: Path
    workspace: CampaignWorkspace
    rounds: str | list[int]  # "unspecified" | "latest" | "all" | [ints]
    data_paths: dict[str, Path]  # built-ins + YAML data entries (resolved)
    output_dir: Path
    filename: str  # final filename (with round suffix applied)
    dpi: int
    format: str  # "png" (default) | "svg" | "pdf"
    logger: logging.Logger
    save_data: bool  # if true, plugins should save tidy data

    def save_df(self, df: pd.DataFrame, filename: str | None = None) -> Path:
        """
        Save a CSV next to the plot (same basename by default).
        Plugins SHOULD check `self.save_data` before calling this.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        target = self.output_dir / (filename or (self.filename.rsplit(".", 1)[0] + ".csv"))
        df.to_csv(target, index=False)
        return target
