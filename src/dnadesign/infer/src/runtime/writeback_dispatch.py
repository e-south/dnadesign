"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/infer/src/runtime/writeback_dispatch.py

Dispatches final extract write-back by ingest source with explicit contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..errors import WriteBackError
from ..writers.pt_file import write_back_pt_file
from ..writers.records import write_back_records


def run_extract_write_back(
    *,
    write_back: bool,
    source: str,
    records,
    pt_path: Optional[str],
    ds,
    ids: Optional[List[str]],
    model_id: str,
    job_id: str,
    columnar: Dict[str, List[object]],
    overwrite: bool,
) -> None:
    if not write_back:
        return

    if source == "records":
        write_back_records(
            records,
            model_id=model_id,
            job_id=job_id,
            columnar=columnar,
            overwrite=overwrite,
        )
        return

    if source == "pt_file":
        write_back_pt_file(
            pt_path,
            records,
            model_id=model_id,
            job_id=job_id,
            columnar=columnar,
            overwrite=overwrite,
        )
        return

    if source == "usr":
        if ids is None or ds is None:
            raise WriteBackError("USR write-back requires ids and dataset handle")
        return

    raise WriteBackError("write_back not supported for this ingest source")
