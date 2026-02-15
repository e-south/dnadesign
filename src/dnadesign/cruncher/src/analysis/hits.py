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
    "best_core_offset",
    "best_core_seq",
    "best_score_norm",
    "best_score_raw",
    "best_score_scaled",
    "best_start",
    "best_strand",
    "best_window_seq",
    "core_def_hash",
    "core_width",
    "draw_idx",
    "elite_id",
    "pwm_hash",
    "pwm_ref",
    "pwm_width",
    "rank",
    "tf",
    "tiebreak_rule",
}

REQUIRED_BASELINE_HITS_COLUMNS = {
    "baseline_id",
    "tf",
    "best_start",
    "best_core_offset",
    "best_strand",
    "best_window_seq",
    "best_core_seq",
    "best_score_raw",
    "best_score_scaled",
    "best_score_norm",
    "tiebreak_rule",
    "pwm_ref",
    "pwm_hash",
    "pwm_width",
    "core_width",
    "core_def_hash",
}


def validate_elites_hits_df(df: pd.DataFrame) -> None:
    missing = [col for col in sorted(REQUIRED_HITS_COLUMNS) if col not in df.columns]
    if missing:
        raise ValueError(f"elites_hits.parquet missing required columns: {missing}")


def load_elites_hits(path: Path) -> pd.DataFrame:
    df = read_parquet(path)
    validate_elites_hits_df(df)
    return df


def validate_baseline_hits_df(df: pd.DataFrame) -> None:
    missing = [col for col in sorted(REQUIRED_BASELINE_HITS_COLUMNS) if col not in df.columns]
    if missing:
        raise ValueError(f"random_baseline_hits.parquet missing required columns: {missing}")


def load_baseline_hits(path: Path) -> pd.DataFrame:
    df = read_parquet(path)
    validate_baseline_hits_df(df)
    return df
