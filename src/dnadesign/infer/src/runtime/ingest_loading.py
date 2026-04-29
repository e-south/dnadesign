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
from ..features.sequence_views import bundle_uses_sequence_views, load_sequence_view_input_records
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
    source_kind: str


def load_extract_ingest(inputs, *, ingest, feature_bundle=None) -> ExtractIngestPayload:
    if feature_bundle is not None and bundle_uses_sequence_views(feature_bundle):
        records = load_sequence_view_input_records(bundle=feature_bundle)
        return ExtractIngestPayload(
            seqs=[str(record["sequence"]) for record in records],
            ids=None,
            records=records,
            pt_path=None,
            dataset=None,
            source_kind="records",
        )
    source = ingest.source
    if source == "sequences":
        seqs = load_sequences_input(inputs)
        return ExtractIngestPayload(seqs=seqs, ids=None, records=None, pt_path=None, dataset=None, source_kind=source)
    if source == "records":
        seqs, records = load_records_input(inputs, ingest.field or "sequence")
        return ExtractIngestPayload(
            seqs=seqs,
            ids=None,
            records=records,
            pt_path=None,
            dataset=None,
            source_kind=source,
        )
    if source == "pt_file":
        if not isinstance(inputs, str):
            raise ValidationError("inputs must be a path string for pt_file ingest")
        seqs, records = load_pt_file_input(inputs, ingest.field or "sequence")
        return ExtractIngestPayload(
            seqs=seqs,
            ids=None,
            records=records,
            pt_path=inputs,
            dataset=None,
            source_kind=source,
        )
    if source == "usr":
        seqs, ids, ds = load_usr_input(
            dataset_name=ingest.dataset,  # type: ignore[arg-type]
            field=ingest.field or "sequence",
            root=ingest.root,
            ids=ingest.ids,
        )
        return ExtractIngestPayload(
            seqs=seqs,
            ids=ids,
            records=None,
            pt_path=None,
            dataset=ds,
            source_kind=source,
        )
    raise ConfigError(f"Unknown ingest source: {source}")


def load_generate_ingest(inputs, *, ingest) -> List[str]:
    payload = load_extract_ingest(inputs, ingest=ingest)
    return payload.seqs
