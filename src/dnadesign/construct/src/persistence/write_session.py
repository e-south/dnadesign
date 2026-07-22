"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/persistence/write_session.py

Construct USR write-session and sequence-view persistence contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Mapping

from dnadesign.usr import Dataset, SequenceViewRecord, load_sequence_view_index, write_sequence_views

from ..contracts.errors import ValidationError
from .records import BuiltRecord, output_records_for_overlay, unique_records_by_output_id
from .usr_registry import (
    _construct_metadata_table,
    _derived_metadata_table,
    _ensure_construct_registry,
    _usr_label_table,
)


def ensure_output_dataset(*, output_root: Path, output_dataset: str) -> Dataset:
    _ensure_construct_registry(output_root)
    return Dataset(output_root, output_dataset)


def construct_actor(job_id: str) -> dict[str, object]:
    run_id = str(os.getenv("USR_ACTOR_RUN_ID") or "").strip() or f"construct-{job_id}"
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
) -> None:
    actor = construct_actor(job_id)
    unique_records = unique_records_by_output_id(records)
    overlay_records = output_records_for_overlay(records)
    with output_ds.write_session() as session:
        session.init_if_missing(
            source="construct",
            notes=f"Initialized by construct job {job_id}.",
            actor=actor,
        )
        if not unique_records:
            return
        source = record_source or f"construct run {job_id}"
        session.import_rows(
            [
                {
                    "sequence": record.sequence,
                    "bio_type": "dna",
                    "alphabet": record.alphabet,
                    "source": source,
                }
                for record in unique_records
            ],
            default_bio_type="dna",
            source=source,
            actor=actor,
        )
        if overlay_records:
            session.write_overlay(
                "construct",
                _construct_metadata_table([record.metadata for record in overlay_records]),
                key="id",
                overwrite=True,
                note="dnadesign.construct lineage attach",
                actor=actor,
            )
        derived_rows = [record.derived_metadata for record in overlay_records if record.derived_metadata is not None]
        if derived_rows:
            session.write_overlay(
                "derived",
                _derived_metadata_table(derived_rows),
                key="id",
                overwrite=True,
                note="dnadesign.construct derived-product attach",
                actor=actor,
            )
        label_rows = [
            {
                "id": record.output_id,
                "usr_label__primary": record.label_primary,
                "usr_label__aliases": record.label_aliases,
            }
            for record in overlay_records
            if record.label_primary is not None or record.label_aliases
        ]
        if label_rows:
            session.write_overlay(
                "usr_label",
                _usr_label_table(label_rows),
                overwrite=True,
                note="dnadesign.construct upstream label carry-through",
                actor=actor,
            )


def sequence_views_to_append(
    sequence_views: list[SequenceViewRecord],
    *,
    existing_by_id: Mapping[str, Mapping[str, object]],
) -> list[SequenceViewRecord]:
    missing_sequence_views: list[SequenceViewRecord] = []
    for view in sequence_views:
        existing = existing_by_id.get(str(view.view_id))
        if existing is None:
            if _has_legacy_or_equivalent_sequence_view(view, existing_by_id=existing_by_id):
                continue
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


def _has_legacy_or_equivalent_sequence_view(
    view: SequenceViewRecord,
    *,
    existing_by_id: Mapping[str, Mapping[str, object]],
) -> bool:
    planned = view.model_dump(mode="python")
    planned.pop("created_at", None)
    planned.pop("created_by", None)
    planned_without_view_id = dict(planned)
    planned_without_view_id.pop("view_id", None)
    for existing in existing_by_id.values():
        if str(existing.get("sequence_id") or "") != str(view.sequence_id):
            continue
        if "product_kind" not in existing or "orientation" not in existing or "view_name" not in existing:
            return True
        existing_without_view_id = dict(existing)
        existing_without_view_id.pop("view_id", None)
        if existing_without_view_id == planned_without_view_id:
            return True
        equivalent_view_fields = (
            "view_name",
            "product_kind",
            "context_kind",
            "orientation",
            "anchor_start_0",
            "anchor_end_0",
            "forward_anchor_start_0",
            "forward_anchor_end_0",
            "recommended_pooling",
        )
        if all(existing.get(field) == planned.get(field) for field in equivalent_view_fields):
            return True
    return False


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
