"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/adapters/sources/sequence_library.py

Sequence library input source (CSV/Parquet).

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .base import BaseDataSource, infer_format, resolve_path


@dataclass
class SequenceLibraryDataSource(BaseDataSource):
    path: str
    cfg_path: Path
    fmt: Optional[str] = None
    sequence_column: str = "sequence"

    def _resolve_format(self, path: Path) -> str:
        fmt = (self.fmt or "").strip().lower() if self.fmt is not None else None
        if fmt:
            if fmt not in {"csv", "parquet"}:
                raise ValueError(f"sequence_library.format must be 'csv' or 'parquet', got: {fmt!r}")
            return fmt
        inferred = infer_format(path)
        if inferred is None:
            raise ValueError(
                f"sequence_library.format is required when file extension is not .csv/.parquet. Got path: {path}"
            )
        return inferred

    def _load_table(self, path: Path, fmt: str) -> pd.DataFrame:
        if fmt == "csv":
            return pd.read_csv(path)
        if fmt == "parquet":
            import pyarrow.parquet as pq

            return pq.read_table(path).to_pandas()
        raise ValueError(f"Unsupported sequence_library.format: {fmt}")

    def load_data(self, *, rng=None, outputs_root: Path | None = None, run_id: str | None = None):
        data_path = resolve_path(self.cfg_path, self.path)
        if not (data_path.exists() and data_path.is_file()):
            raise FileNotFoundError(f"Sequence library file not found. Looked here:\n  - {data_path}")

        fmt = self._resolve_format(data_path)
        df = self._load_table(data_path, fmt)

        if self.sequence_column not in df.columns:
            raise ValueError(f"Column '{self.sequence_column}' missing in {data_path}")
        col = df[self.sequence_column]
        if col.isna().any():
            bad_rows = df[col.isna()].index.tolist()
            preview = ", ".join(str(i) for i in bad_rows[:10])
            raise ValueError(
                f"Null sequences in {data_path} (rows: {preview}). DenseGen requires non-empty sequence strings."
            )
        seqs = [str(s).strip().upper() for s in col.tolist()]
        empty_rows = [i for i, s in enumerate(seqs) if not s]
        if empty_rows:
            preview = ", ".join(str(i) for i in empty_rows[:10])
            raise ValueError(
                f"Empty sequences in {data_path} (rows: {preview}). DenseGen requires non-empty sequence strings."
            )
        invalid_rows = [i for i, s in enumerate(seqs) if any(ch not in {"A", "C", "G", "T"} for ch in s)]
        if invalid_rows:
            preview = ", ".join(str(i) for i in invalid_rows[:10])
            raise ValueError(f"Invalid sequences in {data_path} (rows: {preview}). DenseGen requires A/C/G/T only.")
        return seqs, None, None
