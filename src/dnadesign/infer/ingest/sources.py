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

from pathlib import Path
from typing import Dict, List, Tuple

import torch

from ..errors import ValidationError
from ..logging import get_logger

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
    """
    Default to the monorepo layout:
      <repo>/src/dnadesign/usr/datasets
    """
    here = Path(__file__).resolve()
    # .../dnadesign/infer/ingest/sources.py → parents[2] == .../dnadesign
    root = here.parents[2] / "usr" / "datasets"
    return root


def load_usr_input(
    *,
    dataset_name: str,
    field: str = "sequence",
    root: str | Path | None = None,
    ids: List[str] | None = None,
):
    """
    Load sequences (and aligned ids) from a USR dataset (records.parquet).

    Returns:
      seqs : list[str]
      ids  : list[str]          (same length/order as seqs)
      ds   : dnadesign.usr.Dataset   (opened handle for write-back)
    """
    try:
        from dnadesign.usr import Dataset  # local package, same repo
    except Exception as e:  # pragma: no cover
        raise ValidationError(
            "dnadesign.usr is not importable. Is dnadesign installed in editable mode?"
        ) from e

    ds_root = Path(root) if root else _default_usr_root()
    ds = Dataset(ds_root, dataset_name)

    rec_path = ds.records_path
    if not rec_path.exists():
        raise ValidationError(f"USR dataset not initialized or missing: {rec_path}")

    import pyarrow.parquet as pq

    # read only id + field for efficiency
    tbl = pq.read_table(rec_path, columns=["id", field])

    # Arrow → Python lists
    all_ids = tbl.column("id").to_pylist()
    all_seqs = tbl.column(field).to_pylist()

    if ids:
        # Subset preserving requested order (skip unknowns gracefully)
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

    # Basic checks
    if not seqs:
        raise ValidationError("No sequences found for the requested USR dataset/ids.")
    bad = [i for i, s in enumerate(seqs) if not isinstance(s, str) or not s]
    if bad:
        raise ValidationError(
            f"{len(bad)} empty/invalid sequences found in USR ingest."
        )

    return seqs, ids, ds
