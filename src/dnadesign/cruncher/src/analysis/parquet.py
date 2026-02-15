"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/parquet.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.artifacts.atomic_write import atomic_write_parquet


def read_parquet(path: Path):
    import pandas as pd

    return pd.read_parquet(path, engine="fastparquet")


def write_parquet(df, path: Path) -> None:
    atomic_write_parquet(df, path, engine="pyarrow", index=False)
