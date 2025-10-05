"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/infer/ingest/sources.py

Public API:
  - run_extract
  - run_generate
  - run_job (YAML-driven)

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from .._logging import get_logger
from ..errors import ValidationError

_LOG = get_logger(__name__)


def load_sequences_input(inputs) -> List[str]:
    if isinstance(inputs, str):
        return [inputs]
    if isinstance(inputs, list) and all(isinstance(x, str) for x in inputs):
        return inputs
    raise ValidationError(
        "For ingest.source=sequences, inputs must be str or list[str]"
    )


def load_records_input(inputs, field: str) -> Tuple[List[str], List[Dict]]:
    if not isinstance(inputs, list) or not all(isinstance(x, dict) for x in inputs):
        raise ValidationError("For ingest.source=records, inputs must be list[dict]")
    seqs = []
    for idx, rec in enumerate(inputs):
        if field not in rec:
            raise ValidationError(f"Record at index {idx} missing field '{field}'")
        seqs.append(rec[field])
    return seqs, inputs


def load_records_jsonl_input(path: str, field: str) -> Tuple[List[str], List[Dict]]:
    p = Path(path)
    if not p.exists():
        raise ValidationError(f"records_jsonl not found: {path}")
    data: List[Dict] = []
    with p.open() as f:
        for i, ln in enumerate(f, start=1):
            if not ln.strip():
                continue
            try:
                data.append(json.loads(ln))
            except Exception as e:
                raise ValidationError(f"Invalid JSONL at line {i}: {e}")
    return load_records_input(data, field)


def load_pt_file_input(path: str, field: str) -> Tuple[List[str], List[Dict]]:
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, list) or not all(isinstance(x, dict) for x in data):
        raise ValidationError(".pt file must contain list[dict]")
    seqs = []
    for idx, rec in enumerate(data):
        if field not in rec:
            raise ValidationError(f"Entry {idx} missing field '{field}'")
        seqs.append(rec[field])
    return seqs, data


def _default_usr_root() -> Path:
    """Default to env var DNADESIGN_USR_ROOT if set, else repo layout."""
    env = os.environ.get("DNADESIGN_USR_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    return here.parents[2] / "usr" / "datasets"


def load_usr_input(
    *,
    dataset_name: str,
    field: str = "sequence",
    root: str | Path | None = None,
    ids: List[str] | None = None,
):
    try:
        from dnadesign.usr import Dataset  # local package, same repo
    except Exception as e:
        raise ValidationError(
            "dnadesign.usr is not importable. Is dnadesign installed in editable mode?"
        ) from e

    ds_root = Path(root) if root else _default_usr_root()
    ds = Dataset(ds_root, dataset_name)

    rec_path = ds.records_path
    if not rec_path.exists():
        raise ValidationError(f"USR dataset not initialized or missing: {rec_path}")

    import pyarrow.parquet as pq

    tbl = pq.read_table(rec_path, columns=["id", field])
    all_ids = tbl.column("id").to_pylist()
    all_seqs = tbl.column(field).to_pylist()

    if ids:
        pos = {rid: i for i, rid in enumerate(all_ids)}
        sub_ids, sub_seqs = [], []
        for rid in ids:
            idx = pos.get(rid)
            if idx is not None:
                sub_ids.append(rid)
                sub_seqs.append(all_seqs[idx])
        ids, seqs = sub_ids, sub_seqs
    else:
        ids, seqs = all_ids, all_seqs

    if not seqs:
        raise ValidationError("No sequences found for the requested USR dataset/ids.")
    bad = [i for i, s in enumerate(seqs) if not isinstance(s, str) or not s]
    if bad:
        raise ValidationError(
            f"{len(bad)} empty/invalid sequences found in USR ingest."
        )
    return seqs, ids, ds
