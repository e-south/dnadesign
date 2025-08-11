"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/infer/writers/pt_file.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import time
from typing import Dict, List

import torch
import yaml

from ..errors import WriteBackError
from .records import write_back_records


def write_back_pt_file(
    path: str,
    records: List[dict],
    *,
    model_id: str,
    job_id: str,
    columnar: Dict[str, List[object]],
    overwrite: bool,
) -> None:
    write_back_records(
        records,
        model_id=model_id,
        job_id=job_id,
        columnar=columnar,
        overwrite=overwrite,
    )
    try:
        torch.save(records, path)
    except Exception as e:
        raise WriteBackError(f"torch.save failed for {path}: {e}")


def write_checkpoint(
    dirpath: str,
    *,
    job_id: str,
    model_id: str,
    processed: int,
    total: int,
    extracted_ids: List[str],
) -> str:
    ts = time.strftime("%Y%m%dT%H%M%S")
    progress_path = f"{dirpath}/progress_{job_id}_{ts}.yaml"
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "job_id": job_id,
        "model_id": model_id,
        "processed": processed,
        "total": total,
        "extracted": extracted_ids,
    }
    with open(progress_path, "w") as f:
        yaml.safe_dump(payload, f)
    return progress_path
