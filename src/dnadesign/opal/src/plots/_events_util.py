"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/plots/_events_util.py

Utilities for reading the **ledger** sinks and resolving setpoints for plots.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional, Set, Union

from ..analysis.facade import load_predictions_with_setpoint, read_predictions
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
) -> pd.DataFrame:
    """
    Read the minimum columns needed for a plot **from the ledger** and join
    the setpoint from `ledger.runs` via `objective__params.setpoint_vector`.
    `outputs_dir` should point to the campaign's outputs/ directory.
    """
    maybe_install_pyarrow_sysctl_filter()
    want: Set[str] = set(map(str, base_columns)) | {"run_id"}
    df = load_predictions_with_setpoint(outputs_dir, want, round_selector=round_selector)
    return df.to_pandas()


def load_events(
    outputs_dir: Path,
    base_columns: Iterable[str],
    round_selector: Optional[Union[str, int, List[int]]] = None,
    *,
    allow_missing: bool = False,
) -> pd.DataFrame:
    """
    Read the minimum columns needed for a plot **from the ledger** without
    joining setpoint metadata. Useful for plots that do not require
    objective__params.setpoint_vector.
    """
    maybe_install_pyarrow_sysctl_filter()
    want: Set[str] = set(map(str, base_columns))
    df = read_predictions(
        outputs_dir / "ledger.predictions",
        columns=sorted(want),
        round_selector=round_selector,
        allow_missing=allow_missing,
    )
    return df.to_pandas()
