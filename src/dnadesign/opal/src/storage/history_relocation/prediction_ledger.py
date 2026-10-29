"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/storage/history_relocation/prediction_ledger.py

Selects and materializes exact run identities from OPAL prediction ledgers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from ...core.utils import ExitCodes, OpalError
from ..parquet_io import read_parquet_df

_IDENTITY_COLUMNS = ("as_of_round", "run_id")


def prediction_rows_for_run(
    parts: Iterable[Path],
    *,
    round_index: int,
    run_id: str,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read only rows belonging to one immutable prediction run identity."""

    requested = None if columns is None else list(dict.fromkeys((*columns, *_IDENTITY_COLUMNS)))
    frames: list[pd.DataFrame] = []
    for part in parts:
        frame = read_parquet_df(part, columns=requested)
        matching = frame.loc[frame["as_of_round"].astype(int).eq(round_index) & frame["run_id"].astype(str).eq(run_id)]
        if matching.empty:
            raise OpalError(
                f"Prediction part has no rows for round={round_index}, run_id={run_id}: {part}",
                ExitCodes.CONTRACT_VIOLATION,
            )
        frames.append(matching)
    if not frames:
        raise OpalError(
            f"Prediction ledger has no parts for round={round_index}, run_id={run_id}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    result = pd.concat(frames, ignore_index=True)
    return result if columns is None else result.loc[:, list(columns)]


def prediction_part_is_run_specific(path: Path, *, round_index: int, run_id: str) -> bool:
    """Return whether every row in a prediction part has one requested identity."""

    frame = read_parquet_df(path, columns=list(_IDENTITY_COLUMNS))
    identities = set(zip(frame["as_of_round"].astype(int), frame["run_id"].astype(str), strict=True))
    return identities == {(round_index, run_id)}


__all__ = ["prediction_part_is_run_specific", "prediction_rows_for_run"]
