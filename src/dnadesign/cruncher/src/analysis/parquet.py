"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/parquet.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def read_parquet(path: Path):
    import pandas as pd

    return pd.read_parquet(path, engine="fastparquet")


def write_parquet(df, path: Path) -> None:
    df.to_parquet(path, engine="pyarrow", index=False)
