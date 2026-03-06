"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/infer/src/runtime/extract_chunk_writeback.py

Builds extract chunk write-back callbacks with explicit USR contract checks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Callable, List, Optional

from ..errors import WriteBackError
from ..writers.usr import write_back_usr


def build_extract_chunk_write_back(
    *,
    source: str,
    write_back: bool,
    ds,
    ids: Optional[List[str]],
    model_id: str,
    job_id: str,
    out_id: str,
    overwrite: bool,
    writer: Callable[..., None] = write_back_usr,
) -> Optional[Callable[[List[int], List[object]], None]]:
    if source != "usr" or not write_back:
        return None
    if ids is None or ds is None:
        raise WriteBackError("USR chunk write-back requires ids and dataset handle")

    def _write_back_chunk(idx_chunk: List[int], vals: List[object]) -> None:
        chunk_ids = [ids[row_index] for row_index in idx_chunk]
        writer(
            ds,
            ids=chunk_ids,
            model_id=model_id,
            job_id=job_id,
            columnar={out_id: vals},
            overwrite=overwrite,
        )

    return _write_back_chunk
