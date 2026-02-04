"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/infer/writers/records.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, List

from ..errors import WriteBackError


def write_back_records(
    records: List[dict],
    *,
    model_id: str,
    job_id: str,
    columnar: Dict[str, List[object]],
    overwrite: bool,
) -> None:
    N = len(next(iter(columnar.values()))) if columnar else 0
    if len(records) != N:
        raise WriteBackError("Record length and outputs length mismatch")

    for out_id, col in columnar.items():
        key = f"{model_id}__{job_id}__{out_id}"
        for i, val in enumerate(col):
            if not overwrite and key in records[i]:
                continue
            records[i][key] = val
