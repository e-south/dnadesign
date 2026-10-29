"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/persistence/completion.py

Plan missing Construct metadata from effective SeqLedger rows without overwrites.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import replace

import pyarrow as pa

from dnadesign.seqledger import Dataset, SequenceViewRecord

from ..contracts.errors import ValidationError
from .records import BuiltRecord, output_records_for_overlay
from .usr_registry import _construct_metadata_table, _derived_metadata_table, _usr_label_table


def existing_output_rows(dataset: Dataset, records: list[BuiltRecord]) -> dict[str, dict[str, object]]:
    """Read owned fields through the effective reader while the caller holds its write session."""
    ids = {record.output_id for record in records}
    columns = [
        name
        for name in dataset.schema().names
        if name in {"id", "sequence", "alphabet", "bio_type"}
        or name.startswith(("construct__", "derived__", "usr_label__"))
    ]
    return {
        row["id"]: row
        for batch in dataset.scan(columns=columns, include_deleted=True)
        for row in batch.to_pylist()
        if row["id"] in ids
    }


def preserve_retry_provenance(records: list[BuiltRecord], existing: dict[str, dict[str, object]]) -> list[BuiltRecord]:
    """Preserve an error-policy attempt's identity when its only config change is retry permission."""
    reconciled = []
    for record in records:
        previous_spec = existing.get(record.output_id, {}).get("construct__spec_id")
        if previous_spec is None or previous_spec != record.retry_spec_id:
            reconciled.append(record)
            continue
        derived = record.derived_metadata
        view = record.sequence_view
        if view is not None:
            payload = view.model_dump(mode="python")
            payload.pop("view_id")
            payload.update(derivation_spec_id=previous_spec, derivation_id=f"{record.output_id}:{previous_spec}")
            view = SequenceViewRecord.model_validate(payload)
        reconciled.append(
            replace(
                record,
                metadata={**record.metadata, "construct__spec_id": previous_spec},
                derived_metadata={**derived, "derived__spec_id": previous_spec} if derived is not None else None,
                sequence_view=view,
            )
        )
    return reconciled


def missing_metadata(records: list[BuiltRecord], existing: dict[str, dict[str, object]]) -> list[tuple[str, pa.Table]]:
    """Validate all planned fields before any write, preserving distinct view provenance."""
    for record in records:
        row = existing.get(record.output_id)
        if row is not None and any(
            row[key] != value
            for key, value in (("sequence", record.sequence), ("alphabet", record.alphabet), ("bio_type", "dna"))
        ):
            raise ValidationError(f"Construct output '{record.output_id}' already exists with different sequence data.")
    overlay_records = []
    for record in output_records_for_overlay(records):
        previous_spec = existing.get(record.output_id, {}).get("construct__spec_id")
        if previous_spec is not None and previous_spec != record.metadata["construct__spec_id"]:
            if record.sequence_view is not None:
                # A distinct view owns this derivation; the sequence row keeps its original provenance.
                continue
            raise ValidationError(f"Construct output '{record.output_id}' already exists with different metadata.")
        overlay_records.append(record)
    tables = [("construct", _construct_metadata_table([record.metadata for record in overlay_records]))]
    derived = [record.derived_metadata for record in overlay_records if record.derived_metadata is not None]
    if derived:
        tables.append(("derived", _derived_metadata_table(derived)))
    labels = [
        {"id": record.output_id, "usr_label__primary": record.label_primary, "usr_label__aliases": record.label_aliases}
        for record in overlay_records
        if record.label_primary is not None or record.label_aliases
    ]
    if labels:
        tables.append(("usr_label", _usr_label_table(labels)))
    pending = []
    for namespace, table in tables:
        missing = []
        for row in table.to_pylist():
            current = existing.get(row["id"], {})
            for name, value in row.items():
                if current.get(name) is not None and current[name] != value:
                    raise ValidationError(
                        f"Construct output '{row['id']}' already exists with different metadata in '{name}'."
                    )
            if any(current.get(name) != value for name, value in row.items()):
                missing.append(row)
        if missing:
            pending.append((namespace, pa.Table.from_pylist(missing, schema=table.schema)))
    return pending
