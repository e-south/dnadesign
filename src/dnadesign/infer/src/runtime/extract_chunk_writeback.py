"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/runtime/extract_chunk_writeback.py

Builds extract chunk write-back callbacks with explicit USR contract checks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
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
) -> Optional[Callable[..., None]]:
    if source != "usr" or not write_back:
        return None
    if ids is None or ds is None:
        raise WriteBackError("USR chunk write-back requires ids and dataset handle")

    def _write_back_chunk(
        idx_chunk: List[int],
        vals: List[object],
        *,
        overwrite_override: bool | None = None,
        progress: Mapping[str, object] | None = None,
    ) -> None:
        chunk_ids = [ids[row_index] for row_index in idx_chunk]
        writer(
            ds,
            ids=chunk_ids,
            model_id=model_id,
            job_id=job_id,
            columnar={out_id: vals},
            overwrite=overwrite if overwrite_override is None else bool(overwrite_override),
            event_args=progress,
        )

    return _write_back_chunk


def build_extract_chunk_group_write_back(
    *,
    source: str,
    write_back: bool,
    ds,
    ids: Optional[List[str]],
    model_id: str,
    job_id: str,
    overwrite: bool,
    writer: Callable[..., None] = write_back_usr,
) -> Optional[Callable[..., None]]:
    if source != "usr" or not write_back:
        return None
    if ids is None or ds is None:
        raise WriteBackError("USR chunk write-back requires ids and dataset handle")

    def _write_back_chunk_group(
        idx_chunk: List[int],
        columnar: Mapping[str, List[object]],
        *,
        overwrite_override: bool | None = None,
        event_args: Mapping[str, object] | None = None,
    ) -> None:
        if not columnar:
            return
        chunk_ids = [ids[row_index] for row_index in idx_chunk]
        writer(
            ds,
            ids=chunk_ids,
            model_id=model_id,
            job_id=job_id,
            columnar=dict(columnar),
            overwrite=overwrite if overwrite_override is None else bool(overwrite_override),
            event_args=event_args,
        )

    return _write_back_chunk_group
