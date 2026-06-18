"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/reporting/predictions.py

Public reporting helper for run-scoped OPAL campaign predictions.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from ..analysis.campaign import CampaignAnalysis
from ..core.utils import ExitCodes, OpalError


def read_campaign_predictions(
    config_path: Path | None,
    *,
    columns: Sequence[str] | None = None,
    round_selector: str | None = "latest",
    run_id: str | None = None,
    require_run_id: bool = True,
) -> pd.DataFrame:
    """Read campaign predictions through OPAL's run-aware ledger contract."""

    analysis = CampaignAnalysis.from_config_path(config_path, allow_dir=True)
    frame = analysis.read_predictions(
        columns=columns,
        round_selector=round_selector,
        run_id=run_id,
        require_run_id=require_run_id,
    )
    if frame.is_empty():
        raise OpalError(
            "outputs/ledger/predictions had zero rows for the selected campaign run.",
            ExitCodes.BAD_ARGS,
        )
    return frame.to_pandas()
