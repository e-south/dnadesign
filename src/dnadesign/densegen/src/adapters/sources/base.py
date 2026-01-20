"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/adapters/sources/base.py

Source abstractions for DenseGen inputs.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from ...config import resolve_relative_path


def resolve_path(cfg_path: Path, given: str) -> Path:
    """
    Resolve a path relative to the config directory (no fallback search).
    """
    return resolve_relative_path(cfg_path, given)


def infer_format(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in {".csv"}:
        return "csv"
    if ext in {".parquet", ".pq"}:
        return "parquet"
    return None


class BaseDataSource(abc.ABC):
    @abc.abstractmethod
    def load_data(self, *, rng=None, outputs_root: Path | None = None) -> Tuple[List, Optional[pd.DataFrame]]:
        """
        Returns:
            (data_entries, meta_df)
            - For binding-site inputs: meta_df is a DataFrame with 'tf' and 'tfbs' columns.
            - For sequence library inputs: data_entries is a list of sequences; meta_df None.
        """
        raise NotImplementedError
