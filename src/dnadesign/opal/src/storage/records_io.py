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
from typing import Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..core.utils import OpalError
from .parquet_io import read_parquet_df, write_parquet_df


@dataclass(frozen=True)
class RecordsIO:
    records_path: Path

    def load(self) -> pd.DataFrame:
        if not self.records_path.exists():
            raise OpalError(f"records.parquet not found: {self.records_path}")
        return read_parquet_df(self.records_path)

    def schema_columns(self) -> list[str]:
        if not self.records_path.exists():
            raise OpalError(f"records.parquet not found: {self.records_path}")
        try:
            return list(pq.ParquetFile(self.records_path).schema_arrow.names)
        except Exception as exc:
            raise OpalError(f"Failed to read records.parquet schema: {self.records_path}: {exc}") from exc

    def row_count(self) -> int:
        if not self.records_path.exists():
            raise OpalError(f"records.parquet not found: {self.records_path}")
        try:
            return int(pq.ParquetFile(self.records_path).metadata.num_rows)
        except Exception as exc:
            raise OpalError(f"Failed to read records.parquet metadata: {self.records_path}: {exc}") from exc

    def load_columns(self, columns: Sequence[str]) -> pd.DataFrame:
        if not self.records_path.exists():
            raise OpalError(f"records.parquet not found: {self.records_path}")
        available = set(self.schema_columns())
        selected = [str(column) for column in dict.fromkeys(columns) if str(column) in available]
        return read_parquet_df(self.records_path, columns=selected)

    def save_atomic(self, df: pd.DataFrame) -> None:
        tmp = self.records_path.with_suffix(".tmp.parquet")
        write_parquet_df(tmp, df, index=False)
        tmp.replace(self.records_path)

    def append_null_column_atomic(self, column_name: str, *, batch_size: int = 512) -> bool:
        if not self.records_path.exists():
            raise OpalError(f"records.parquet not found: {self.records_path}")
        try:
            source = pq.ParquetFile(self.records_path)
        except Exception as exc:
            raise OpalError(f"Failed to read records.parquet schema: {self.records_path}: {exc}") from exc
        if column_name in source.schema_arrow.names:
            return False

        tmp = self.records_path.with_suffix(".tmp.parquet")
        out_schema = source.schema_arrow.append(pa.field(column_name, pa.null()))
        writer = None
        failed = False
        try:
            writer = pq.ParquetWriter(tmp, out_schema)
            for batch in source.iter_batches(batch_size=batch_size):
                writer.write_batch(batch.append_column(column_name, pa.nulls(batch.num_rows, type=pa.null())))
        except Exception as exc:
            failed = True
            raise OpalError(f"Failed to append column {column_name!r} to records.parquet: {exc}") from exc
        finally:
            if writer is not None:
                writer.close()
            if failed and tmp.exists():
                tmp.unlink()
        tmp.replace(self.records_path)
        return True
