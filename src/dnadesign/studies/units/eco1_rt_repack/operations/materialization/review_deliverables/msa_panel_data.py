"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/msa_panel_data.py

Source-record and alignment helpers for Eco1 review MSA panels.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.rendering import (
    shorten_label,
)


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    current_id = ""
    chunks: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if current_id:
                records.append((current_id, "".join(chunks)))
            current_id = line[1:].strip()
            chunks = []
        elif line.strip():
            chunks.append(line.strip())
    if current_id:
        records.append((current_id, "".join(chunks)))
    return records


def source_manifest_accessions(path: Path) -> set[str]:
    """Return accession identifiers from a conservation source manifest."""

    return {
        str(row.get("accession") or "").strip()
        for row in _source_manifest_records(path)
        if str(row.get("accession") or "").strip()
    }


def source_record_labels(path: Path, *, row_label_prefix: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for index, raw_record in enumerate(_source_manifest_records(path)):
        record_id = str(raw_record.get("record_id") or "").strip()
        if not record_id:
            raise ValueError(f"source manifest included_records[{index}].record_id is required")
        accession = str(raw_record.get("accession") or "").strip()
        node = _record_node(record_id)
        label = " ".join(part for part in (row_label_prefix, node, accession) if part)
        labels[record_id] = shorten_label(label or record_id, max_length=52)
    return labels


def source_record_accessions(path: Path) -> dict[str, str]:
    accessions: dict[str, str] = {}
    for index, raw_record in enumerate(_source_manifest_records(path)):
        record_id = str(raw_record.get("record_id") or "").strip()
        if not record_id:
            raise ValueError(f"source manifest included_records[{index}].record_id is required")
        accession = str(raw_record.get("accession") or "").strip()
        if accession:
            accessions[record_id] = accession
    return accessions


def format_msa_row_label(
    record_id: str,
    *,
    source_labels: dict[str, str],
    profile_id: str,
    row_label_prefix: str,
) -> str:
    if record_id == "eco1_rt_ec86kit_reference":
        return "Ec86 reference"
    if record_id in source_labels:
        return source_labels[record_id]
    prefix = f"{profile_id}__"
    if record_id.startswith(prefix):
        parts = [part for part in record_id[len(prefix) :].split("__") if part]
        if parts:
            terminal = parts[-1]
            return f"{row_label_prefix} row {shorten_label(terminal, max_length=14)}"
    if record_id.startswith("clade9_neighbor_"):
        return f"{row_label_prefix} row {record_id.rsplit('_', maxsplit=1)[-1]}"
    return shorten_label(record_id, max_length=28)


def select_display_records(
    records: list[tuple[str, str]],
    profile_rows: list[dict[str, Any]],
    *,
    protected_positions: set[int],
    conserved_positions: set[int],
    max_display_rows: int | None,
) -> list[tuple[str, str]]:
    if max_display_rows is None or len(records) <= max_display_rows:
        return records
    target_record = records[0]
    scored = [
        (
            _display_difference_score(sequence, profile_rows, protected_positions, conserved_positions),
            record_id,
            sequence,
        )
        for record_id, sequence in records[1:]
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [target_record, *[(record_id, sequence) for _score, record_id, sequence in scored[: max_display_rows - 1]]]


def order_selected_records(
    records: list[tuple[str, str]],
    *,
    source_accessions: dict[str, str],
    subtype_accessions: set[str],
) -> list[tuple[str, str]]:
    if not subtype_accessions or len(records) <= 1:
        return records
    reference = [records[0]]
    source_rows = records[1:]
    subtype_rows = [record for record in source_rows if source_accessions.get(record[0]) in subtype_accessions]
    remaining_rows = [record for record in source_rows if source_accessions.get(record[0]) not in subtype_accessions]
    return [*reference, *subtype_rows, *remaining_rows]


def subtype_row_segments(
    records: list[tuple[str, str]],
    *,
    source_accessions: dict[str, str],
    subtype_accessions: set[str],
) -> list[tuple[int, int]]:
    if not subtype_accessions:
        return []
    segments: list[tuple[int, int]] = []
    start: int | None = None
    count = 0
    for index, (record_id, _sequence) in enumerate(records):
        is_subtype = source_accessions.get(record_id) in subtype_accessions
        if is_subtype and start is None:
            start = index
            count = 1
        elif is_subtype:
            count += 1
        elif start is not None:
            segments.append((start, count))
            start = None
            count = 0
    if start is not None:
        segments.append((start, count))
    return segments


def alignment_matrix(records: list[tuple[str, str]], profile_rows: list[dict[str, Any]]) -> list[list[int]]:
    matrix: list[list[int]] = []
    wt_by_index = [str(row["wt_aa"]) for row in profile_rows]
    columns = [int(row["msa_column"]) - 1 for row in profile_rows]
    for _record_id, sequence in records:
        row_values: list[int] = []
        for index, column in enumerate(columns):
            residue = sequence[column] if 0 <= column < len(sequence) else "-"
            if residue == "-":
                row_values.append(0)
            elif residue == wt_by_index[index]:
                row_values.append(2)
            else:
                row_values.append(1)
        matrix.append(row_values)
    return matrix


def _source_manifest_records(path: Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"source manifest must be a YAML mapping: {path}")
    raw_records = payload.get("included_records")
    if not isinstance(raw_records, list):
        raise ValueError(f"source manifest must declare included_records as a list: {path}")
    records: list[dict[str, Any]] = []
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, dict):
            raise ValueError(f"source manifest included_records[{index}] must be a mapping")
        records.append(raw_record)
    return records


def _record_node(record_id: str) -> str:
    if record_id.startswith("clade9_neighbor_"):
        return record_id.rsplit("_", maxsplit=1)[-1]
    parts = record_id.rsplit("__", 2)
    if len(parts) == 3:
        return parts[1]
    return ""


def _display_difference_score(
    sequence: str,
    profile_rows: list[dict[str, Any]],
    protected_positions: set[int],
    conserved_positions: set[int],
) -> int:
    score = 0
    for row in profile_rows:
        position = int(row["canonical_position"])
        column = int(row["msa_column"]) - 1
        residue = sequence[column] if 0 <= column < len(sequence) else "-"
        if residue == str(row["wt_aa"]):
            continue
        score += 1
        if position in conserved_positions:
            score += 3
        if position in protected_positions:
            score += 2
    return score
