"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/api/materialize.py

Public result materialization API.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import pandas as pd

from dnadesign.permuter.src.api.contracts import DatasetRef, PermuterResult, VariantRecord
from dnadesign.permuter.src.contracts.metrics import observed_metric_column, observed_metric_subcolumn
from dnadesign.permuter.src.core.ids import variant_id
from dnadesign.permuter.src.core.storage import (
    append_record_event,
    atomic_write_parquet,
    write_ref_fasta,
    write_ref_protein_fasta,
)
from dnadesign.permuter.src.core.usr import make_usr_row


def materialize_result(result: PermuterResult, output_dir: str | Path, *, overwrite: bool = False) -> DatasetRef:
    """
    Write a public API result as a USR-shaped Permuter dataset directory.

    The materialized dataset intentionally uses the same canonical observed
    metric columns as the CLI so downstream tools can consume either path.
    """

    dataset_dir = Path(output_dir).expanduser().resolve()
    records_path = dataset_dir / "records.parquet"
    if records_path.exists() and not overwrite:
        raise FileExistsError(f"Dataset already exists: {records_path}. Set overwrite=True to replace it.")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    rows = [_record_to_usr_row(result, record) for record in result.records]
    if not rows:
        raise ValueError("PermuterResult.records must contain at least one record to materialize")
    df = pd.DataFrame(rows)
    atomic_write_parquet(df, records_path)
    ref_path = write_ref_fasta(dataset_dir, result.ref_name, result.reference_sequence)
    ref_aa_path = None
    if result.bio_type == "protein":
        ref_aa_path = write_ref_protein_fasta(dataset_dir, result.ref_name, result.reference_sequence)
    record_path = append_record_event(
        dataset_dir,
        "MATERIALIZE",
        [
            f"request_id: {result.request_id}",
            f"ref: {result.ref_name}",
            f"bio_type: {result.bio_type}",
            f"records: {records_path}",
        ],
    )
    return DatasetRef(
        dataset_dir=dataset_dir,
        records_path=records_path,
        row_count=len(df),
        ref_path=ref_path,
        ref_aa_path=ref_aa_path,
        record_path=record_path,
    )


def _record_to_usr_row(result: PermuterResult, record: VariantRecord) -> dict[str, object]:
    if record.ref_name != result.ref_name:
        raise ValueError(
            f"{record.id}: record.ref_name {record.ref_name!r} does not match result ref {result.ref_name!r}"
        )
    if record.bio_type != result.bio_type:
        raise ValueError(
            f"{record.id}: record.bio_type {record.bio_type!r} does not match result bio_type {result.bio_type!r}"
        )
    row = make_usr_row(
        sequence=record.sequence,
        bio_type=record.bio_type,
        source=f"permuter api {result.request_id}",
    )
    if row["id"] != record.id:
        raise ValueError(f"{record.id}: record.id does not match canonical USR id for sequence")
    permuter = _permuter_payload(record)
    protocol = str(permuter.get("protocol") or _result_protocol(result) or "api")
    row.update(
        {
            "permuter__scope": result.request_id,
            "permuter__ref": record.ref_name,
            "permuter__protocol": protocol,
            "permuter__var_id": variant_id(
                scope=result.request_id,
                ref=record.ref_name,
                protocol=protocol,
                sequence=record.sequence,
                modifications=list(record.modifications),
            ),
            "permuter__modifications": list(record.modifications),
            "permuter__round": 1,
        }
    )
    caller_metadata = {key: value for key, value in record.metadata.items() if key != "permuter"}
    if caller_metadata:
        row["permuter__caller_metadata_json"] = json.dumps(caller_metadata, sort_keys=True, default=str)
    for key, value in permuter.items():
        if key in {"protocol", "observed"}:
            continue
        row[f"permuter__{key}"] = _column_value(value)
    _attach_observed_columns(row, permuter.get("observed"))
    return row


def _permuter_payload(record: VariantRecord) -> Mapping[str, object]:
    payload = record.metadata.get("permuter")
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"{record.id}: metadata.permuter must be a mapping")
    return payload


def _result_protocol(result: PermuterResult) -> object:
    payload = result.metadata.get("permuter")
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError("PermuterResult.metadata.permuter must be a mapping")
    return payload.get("protocol")


def _attach_observed_columns(row: dict[str, object], observed: object) -> None:
    if observed is None:
        return
    if not isinstance(observed, Mapping):
        raise ValueError("metadata.permuter.observed must be a mapping")
    for metric_id, value in observed.items():
        if isinstance(value, Mapping):
            for suffix, subvalue in value.items():
                row[observed_metric_subcolumn(str(metric_id), str(suffix))] = _column_value(subvalue)
        else:
            row[observed_metric_column(str(metric_id))] = _column_value(value)


def _column_value(value: object) -> object:
    if isinstance(value, tuple):
        return list(value)
    return value
