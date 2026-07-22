"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/promoter_candidate_bindings/parquet_io.py

Parquet serialization contract for promoter candidate bindings.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .contracts import BINDINGS_RECORD_ID, SCHEMA_ID, SCHEMA_VERSION, STUDY_ID, PromoterCandidateBindingsError
from .row_contract import BINDING_COLUMNS

_METADATA = {
    b"schema_id": SCHEMA_ID.encode(),
    b"schema_version": SCHEMA_VERSION.encode(),
    b"study_id": STUDY_ID.encode(),
    b"record_id": BINDINGS_RECORD_ID.encode(),
}


def write_bindings(rows: pd.DataFrame, path: Path) -> None:
    try:
        table = pa.Table.from_pandas(rows.loc[:, BINDING_COLUMNS], preserve_index=False)
        metadata = dict(table.schema.metadata or {})
        metadata.update(_METADATA)
        pq.write_table(table.replace_schema_metadata(metadata), path)
    except Exception as exc:
        raise PromoterCandidateBindingsError(f"Could not write promoter binding Parquet record: {exc}") from exc


def read_bindings(path: Path) -> pd.DataFrame:
    verify_metadata(path)
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        raise PromoterCandidateBindingsError(f"Could not read promoter binding record {path}: {exc}") from exc


def verify_metadata(path: Path) -> None:
    try:
        metadata = pq.read_metadata(path).metadata or {}
    except Exception as exc:
        raise PromoterCandidateBindingsError(f"Could not read promoter binding Parquet metadata: {exc}") from exc
    actual = {key: metadata.get(key) for key in _METADATA}
    if actual != _METADATA:
        raise PromoterCandidateBindingsError(f"Promoter binding Parquet schema metadata mismatch: {actual}")


__all__ = ["read_bindings", "verify_metadata", "write_bindings"]
