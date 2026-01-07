"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/storage/records_io.py

Module Author(s): Eric J. South (extended by Codex)
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..core.utils import OpalError


@dataclass(frozen=True)
class RecordsIO:
    records_path: Path

    def load(self) -> pd.DataFrame:
        if not self.records_path.exists():
            raise OpalError(f"records.parquet not found: {self.records_path}")
        return pd.read_parquet(self.records_path)

    def save_atomic(self, df: pd.DataFrame) -> None:
        tmp = self.records_path.with_suffix(".tmp.parquet")
        df.to_parquet(tmp, index=False)
        tmp.replace(self.records_path)
