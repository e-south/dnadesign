"""Parquet helpers."""

from __future__ import annotations

from pathlib import Path


def read_parquet(path: Path):
    import pandas as pd

    return pd.read_parquet(path, engine="fastparquet")
