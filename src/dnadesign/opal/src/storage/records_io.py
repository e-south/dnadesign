"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/storage/records_io.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..core.utils import OpalError
from .parquet_io import read_parquet_df, write_parquet_df


@dataclass(frozen=True)
class RecordsIO:
    records_path: Path

    def load(self) -> pd.DataFrame:
        if not self.records_path.exists():
            raise OpalError(f"records.parquet not found: {self.records_path}")
        return read_parquet_df(self.records_path)

    def save_atomic(self, df: pd.DataFrame) -> None:
        tmp = self.records_path.with_suffix(".tmp.parquet")
        write_parquet_df(tmp, df, index=False)
        tmp.replace(self.records_path)
