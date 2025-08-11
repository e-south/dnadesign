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

from typing import Dict, List, Tuple

import torch

from ..errors import ValidationError


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
