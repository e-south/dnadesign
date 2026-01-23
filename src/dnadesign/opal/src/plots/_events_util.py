# ABOUTME: Loads ledger-backed event data for plot plugins.
# ABOUTME: Resolves output paths for predictions and run metadata.
"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/plots/_events_util.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional, Set, Union

from ..analysis.facade import load_predictions_with_setpoint, read_predictions, read_runs
from ..core.stderr_filter import maybe_install_pyarrow_sysctl_filter

if TYPE_CHECKING:
    import pandas as pd


def resolve_outputs_dir(context) -> Path:
    """
    Resolve the campaign outputs/ directory (ledger sinks live here).
    """
    if hasattr(context, "workspace"):
        return Path(context.workspace.outputs_dir)
    return Path(context.campaign_dir) / "outputs"


def load_events_with_setpoint(
    outputs_dir: Path,
    base_columns: Iterable[str],
    round_selector: Optional[Union[str, int, List[int]]] = None,
    *,
    run_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Read the minimum columns needed for a plot **from the ledger** and join
    the setpoint from outputs/ledger/runs.parquet via `objective__params.setpoint_vector`.
    `outputs_dir` should point to the campaign's outputs/ directory.
    If multiple run_ids exist for the selected round(s), run_id is required.
    """
    maybe_install_pyarrow_sysctl_filter()
    want: Set[str] = set(map(str, base_columns)) | {"run_id"}
    df = load_predictions_with_setpoint(outputs_dir, want, round_selector=round_selector, run_id=run_id)
    return df.to_pandas()


def load_events(
    outputs_dir: Path,
    base_columns: Iterable[str],
    round_selector: Optional[Union[str, int, List[int]]] = None,
    *,
    run_id: Optional[str] = None,
    allow_missing: bool = False,
) -> pd.DataFrame:
    """
    Read the minimum columns needed for a plot **from the ledger** without
    joining setpoint metadata. Useful for plots that do not require
    objective__params.setpoint_vector.
    If multiple run_ids exist for the selected round(s), run_id is required.
    """
    maybe_install_pyarrow_sysctl_filter()
    want: Set[str] = set(map(str, base_columns))
    runs_df = read_runs(outputs_dir / "ledger" / "runs.parquet")
    df = read_predictions(
        outputs_dir / "ledger" / "predictions",
        columns=sorted(want),
        round_selector=round_selector,
        run_id=run_id,
        runs_df=runs_df,
        allow_missing=allow_missing,
    )
    return df.to_pandas()
