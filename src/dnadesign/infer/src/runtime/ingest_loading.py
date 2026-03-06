"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/runtime/ingest_loading.py

Runtime ingest loading contracts for extract and generate execution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..errors import ConfigError, ValidationError
from ..ingest.sources import (
    load_pt_file_input,
    load_records_input,
    load_sequences_input,
    load_usr_input,
)


@dataclass(frozen=True)
class ExtractIngestPayload:
    seqs: List[str]
    ids: Optional[List[str]]
    records: Optional[List[Dict[str, Any]]]
    pt_path: Optional[str]
    dataset: object


def load_extract_ingest(inputs, *, ingest) -> ExtractIngestPayload:
    source = ingest.source
    if source == "sequences":
        seqs = load_sequences_input(inputs)
        return ExtractIngestPayload(seqs=seqs, ids=None, records=None, pt_path=None, dataset=None)
    if source == "records":
        seqs, records = load_records_input(inputs, ingest.field or "sequence")
        return ExtractIngestPayload(seqs=seqs, ids=None, records=records, pt_path=None, dataset=None)
    if source == "pt_file":
        if not isinstance(inputs, str):
            raise ValidationError("inputs must be a path string for pt_file ingest")
        seqs, records = load_pt_file_input(inputs, ingest.field or "sequence")
        return ExtractIngestPayload(seqs=seqs, ids=None, records=records, pt_path=inputs, dataset=None)
    if source == "usr":
        seqs, ids, ds = load_usr_input(
            dataset_name=ingest.dataset,  # type: ignore[arg-type]
            field=ingest.field or "sequence",
            root=ingest.root,
            ids=ingest.ids,
        )
        return ExtractIngestPayload(seqs=seqs, ids=ids, records=None, pt_path=None, dataset=ds)
    raise ConfigError(f"Unknown ingest source: {source}")


def load_generate_ingest(inputs, *, ingest) -> List[str]:
    payload = load_extract_ingest(inputs, ingest=ingest)
    return payload.seqs
