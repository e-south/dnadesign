"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/storage/records_io.py

Storage helpers for records IO OPAL storage.

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
from .parquet_io import read_parquet_df


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
        self.records_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            table = self._table_preserving_existing_schema(df)
            pq.write_table(table, tmp)
        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            raise OpalError(f"Failed to write records.parquet atomically: {exc}") from exc
        tmp.replace(self.records_path)

    def _table_preserving_existing_schema(self, df: pd.DataFrame) -> pa.Table:
        inferred = pa.Table.from_pandas(df, preserve_index=False)
        if not self.records_path.exists():
            return inferred
        source_schema = pq.ParquetFile(self.records_path).schema_arrow
        fields: list[pa.Field] = []
        for column in df.columns:
            col = str(column)
            source_index = source_schema.get_field_index(col)
            if source_index >= 0:
                source_field = source_schema.field(source_index)
                # label_hist evolves from label-only to label+prediction entries; infer it from current rows.
                if not pa.types.is_null(source_field.type) and not col.endswith("__label_hist"):
                    fields.append(source_field)
                    continue
            fields.append(inferred.schema.field(col))
        return pa.Table.from_pandas(df, preserve_index=False, schema=pa.schema(fields))

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
