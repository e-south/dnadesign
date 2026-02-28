"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/src/dataset_reporting.py

Reporting and profiling helpers for USR datasets.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from .events import fingerprint_parquet
from .types import Manifest


def manifest_dataset(dataset: Any, *, include_events: bool = False) -> Manifest:
    """Return a structured manifest of the dataset, overlays, and snapshots."""
    dataset._require_exists()
    base_pf = pq.ParquetFile(str(dataset.records_path))
    base_meta = base_pf.schema_arrow.metadata or {}
    meta_decoded = {k.decode("utf-8"): v.decode("utf-8") for k, v in base_meta.items()}
    snapshots = []
    if dataset.snapshot_dir.exists():
        snapshots = sorted(str(p) for p in dataset.snapshot_dir.glob("records-*.parquet"))
    events_count = None
    if include_events and dataset.events_path.exists():
        events_count = sum(1 for _ in dataset.events_path.open("r", encoding="utf-8"))
    return Manifest(
        name=dataset.name,
        path=str(dataset.records_path),
        metadata=meta_decoded,
        fingerprint=fingerprint_parquet(dataset.records_path),
        overlays=dataset.list_overlays(),
        snapshots=snapshots,
        events_count=events_count,
    )


def manifest_dict_dataset(dataset: Any, *, include_events: bool = False) -> dict:
    return manifest_dataset(dataset, include_events=include_events).to_dict()


def describe_dataset(
    dataset: Any,
    opts,
    *,
    columns: Optional[List[str]] = None,
    sample: int = 1024,
    batch_size: int = 65536,
    include_deleted: bool = False,
    tombstone_columns: tuple[str, ...],
) -> List[dict]:
    """Profile columns with optional overlay merge."""
    from .pretty import profile_batches

    merged_schema = dataset.schema()
    cols = columns if columns else list(merged_schema.names)
    if not include_deleted:
        cols = [c for c in cols if c not in tombstone_columns]
    out_schema = pa.schema([merged_schema.field(name) for name in cols])

    return profile_batches(
        dataset.scan(
            columns=cols,
            include_overlays=True,
            include_deleted=include_deleted,
            batch_size=int(batch_size),
        ),
        out_schema,
        opts,
        columns=cols,
        sample=int(sample),
        total_rows=None,
    )
