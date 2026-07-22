"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe/core/selection_artifacts.py

OPAL v3 selection artifact contract for single-view probe campaigns.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dnadesign.opal import read_selection_artifact

PROBE_SELECTION_VIEW_ID = "primary"


def probe_selection_path(workdir: Path, round_index: int) -> Path:
    return workdir / "outputs" / "rounds" / f"round_{int(round_index)}" / "selection" / "selections.parquet"


def read_probe_selection(workdir: Path, round_index: int) -> pd.DataFrame:
    """Read and validate the only selection view emitted by a probe campaign."""

    path = probe_selection_path(workdir, round_index)
    frame = read_selection_artifact(path, required_columns=("id", "selection_view_id"))
    view_ids = set(frame["selection_view_id"].astype(str))
    if view_ids != {PROBE_SELECTION_VIEW_ID}:
        raise ValueError(
            f"Probe selection artifact requires only view {PROBE_SELECTION_VIEW_ID!r}; "
            f"observed {sorted(view_ids)}: {path}"
        )
    return frame.reset_index(drop=True)


__all__ = ["PROBE_SELECTION_VIEW_ID", "probe_selection_path", "read_probe_selection"]
