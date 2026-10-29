"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/persistence/write_session.py

Construct SeqLedger write-session and sequence-view persistence contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Mapping

from dnadesign.seqledger import Dataset, SequenceViewRecord, load_sequence_view_index, write_sequence_views

from ..contracts.errors import ValidationError
from .completion import existing_output_rows, missing_metadata, preserve_retry_provenance
from .records import BuiltRecord, unique_records_by_output_id
from .usr_registry import (
    _ensure_construct_registry,
)


def ensure_output_dataset(*, output_root: Path, output_dataset: str) -> Dataset:
    _ensure_construct_registry(output_root)
    return Dataset(output_root, output_dataset)


def construct_actor(job_id: str) -> dict[str, object]:
    run_id = str(os.getenv("SEQLEDGER_ACTOR_RUN_ID") or "").strip() or f"construct-{job_id}"
    return {
        "tool": "construct",
        "run_id": run_id,
        "host": socket.gethostname(),
        "pid": os.getpid(),
    }


def write_output_records(
    output_ds: Dataset,
    *,
    job_id: str,
    record_source: str | None,
    records: list[BuiltRecord],
    on_conflict: str = "error",
) -> tuple[int, list[BuiltRecord]]:
    actor = construct_actor(job_id)
    unique_records = unique_records_by_output_id(records)
    with output_ds.write_session() as session:
        session.init_if_missing(
            source="construct",
            notes=f"Initialized by construct job {job_id}.",
            actor=actor,
        )
        existing = existing_output_rows(output_ds, unique_records)
        if existing and on_conflict == "error":
            raise ValidationError(f"{len(existing)} planned output id(s) already exist in dataset '{output_ds.name}'.")
        records = preserve_retry_provenance(records, existing)
        pending = missing_metadata(records, existing)
        sequence_views_to_append(
            [record.sequence_view for record in records if record.sequence_view is not None],
            existing_by_id=load_sequence_view_index(output_ds),
        )
        new_records = [record for record in unique_records if record.output_id not in existing]
        if new_records:
            source = record_source or f"construct run {job_id}"
            session.import_rows(
                [
                    {"sequence": record.sequence, "bio_type": "dna", "alphabet": record.alphabet, "source": source}
                    for record in new_records
                ],
                default_bio_type="dna",
                source=source,
                actor=actor,
            )
        for namespace, table in pending:
            session.write_overlay(
                namespace,
                table,
                key="id",
                overwrite=True,
                note=f"dnadesign.construct {namespace} completion",
                actor=actor,
            )
        return sum(record.output_id not in existing for record in records), records


def sequence_views_to_append(
    sequence_views: list[SequenceViewRecord],
    *,
    existing_by_id: Mapping[str, Mapping[str, object]],
) -> list[SequenceViewRecord]:
    missing_sequence_views: list[SequenceViewRecord] = []
    for view in sequence_views:
        existing = existing_by_id.get(str(view.view_id))
        if existing is None:
            missing_sequence_views.append(view)
            continue
        comparable_view = view.model_dump(mode="python")
        comparable_view.pop("created_at", None)
        comparable_view.pop("created_by", None)
        if existing != comparable_view:
            raise ValidationError(
                f"Sequence view '{view.view_id}' already exists with different metadata; "
                "refusing to treat the rerun as idempotent."
            )
    return missing_sequence_views


def write_planned_sequence_views(output_ds: Dataset, *, job_id: str, records: list[BuiltRecord]) -> None:
    sequence_views = [record.sequence_view for record in records if record.sequence_view is not None]
    if not sequence_views:
        return
    missing_sequence_views = sequence_views_to_append(
        sequence_views,
        existing_by_id=load_sequence_view_index(output_ds),
    )
    if missing_sequence_views:
        write_sequence_views(
            output_ds,
            [view.model_dump(mode="python") for view in missing_sequence_views],
            conflict_policy="idempotent",
            actor=construct_actor(job_id),
        )
