"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/runtime/run_construct_helpers.py

Shared fixtures for construct runtime realization tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from dnadesign.usr.src.registry.models import SEQ_ANNOT_COLUMNS
from dnadesign.usr.src.registry.typespec import arrow_type_from_str


def write_registry(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "registry.yaml").write_text(
        """
namespaces:
  usr_state:
    owner: usr
    description: Reserved record-state overlay (masked/qc/split/lineage).
    columns:
      - name: usr_state__masked
        type: bool
      - name: usr_state__qc_status
        type: string
      - name: usr_state__split
        type: string
      - name: usr_state__supersedes
        type: string
      - name: usr_state__lineage
        type: list<string>
""",
        encoding="utf-8",
    )


def seq_annot_table(*, row_id: str, features: list[dict[str, object]]) -> pa.Table:
    seq_annot_type = next(column.type for column in SEQ_ANNOT_COLUMNS if column.name == "seq_annot__features")
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("seq_annot__features", arrow_type_from_str(seq_annot_type)),
        ]
    )
    return pa.table(
        {
            "id": pa.array([row_id], type=pa.string()),
            "seq_annot__features": pa.array([features], type=arrow_type_from_str(seq_annot_type)),
        },
        schema=schema,
    )


__all__ = ["seq_annot_table", "write_registry"]
