"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/dataset_read_keys.py

Record-batch key extraction helpers for dataset read and dedupe operations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List

import pyarrow as pa

from .errors import SchemaError


def key_list_from_batch(batch: pa.RecordBatch, key: str) -> List[str]:
    def _col(name: str) -> pa.Array:
        idx = batch.schema.get_field_index(name)
        if idx < 0:
            raise SchemaError(f"Missing required column '{name}' for key '{key}'.")
        return batch.column(idx)

    if key == "id":
        vals = _col("id").to_pylist()
        if any(v is None or str(v).strip() == "" for v in vals):
            raise SchemaError("Missing id values while computing dedupe key.")
        return [str(v) for v in vals]
    if key in {"sequence", "sequence_norm"}:
        vals = _col("sequence").to_pylist()
        if any(v is None or str(v).strip() == "" for v in vals):
            raise SchemaError("Missing sequence values while computing dedupe key.")
        return [str(v).strip() for v in vals]
    if key == "sequence_ci":
        alph = _col("alphabet").to_pylist()
        if any(v is None or str(v).strip() == "" for v in alph):
            raise SchemaError("Missing alphabet values while computing dedupe key.")
        if any(str(v) != "dna_4" for v in alph):
            raise SchemaError("sequence_ci is only valid for dna_4 datasets.")
        seqs = _col("sequence").to_pylist()
        if any(v is None or str(v).strip() == "" for v in seqs):
            raise SchemaError("Missing sequence values while computing dedupe key.")
        return [str(v).strip().upper() for v in seqs]
    raise SchemaError(f"Unsupported join key '{key}'.")
