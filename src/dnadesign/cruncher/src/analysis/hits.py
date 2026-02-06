"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/hits.py

Load and validate elite best-hit metadata.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dnadesign.cruncher.analysis.parquet import read_parquet

REQUIRED_HITS_COLUMNS = {
    "elite_id",
    "tf",
    "best_start",
    "best_strand",
    "best_core_seq",
    "pwm_width",
}


def validate_elites_hits_df(df: pd.DataFrame) -> None:
    missing = [col for col in sorted(REQUIRED_HITS_COLUMNS) if col not in df.columns]
    if missing:
        raise ValueError(f"elites_hits.parquet missing required columns: {missing}")


def load_elites_hits(path: Path) -> pd.DataFrame:
    df = read_parquet(path)
    validate_elites_hits_df(df)
    return df
