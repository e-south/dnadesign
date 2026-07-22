"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/structure_views/_mmcif.py

Fail-safe helpers for reading mmCIF coordinate loops.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class MmcifAtomSiteRecord:
    """Identity fields and source lines for one parsed atom-site record."""

    atom_name: str
    residue_name: str
    chain_id: str
    residue_number: str
    insertion_code: str
    source_line_indices: tuple[int, ...]


def iter_mmcif_atom_site_records(structure_text: str) -> Iterator[MmcifAtomSiteRecord]:
    """Yield normalized identity fields from declared mmCIF atom-site loops."""

    lines = structure_text.splitlines()
    line_index = 0
    while line_index < len(lines):
        if lines[line_index].strip().lower() != "loop_":
            line_index += 1
            continue

        line_index += 1
        headers: list[str] = []
        while line_index < len(lines):
            stripped = lines[line_index].strip()
            if not stripped.startswith("_"):
                break
            headers.append(stripped.split(maxsplit=1)[0].lower())
            line_index += 1

        if not headers or not all(header.startswith("_atom_site.") for header in headers):
            line_index = _skip_loop_values(lines, line_index)
            continue

        field_indices = _atom_site_field_indices(headers)
        if field_indices is None:
            line_index = _skip_loop_values(lines, line_index)
            continue

        value_buffer: list[tuple[str, int]] = []
        while line_index < len(lines):
            stripped = lines[line_index].strip()
            if _is_loop_boundary(stripped):
                break
            source_line_index = line_index
            line_index += 1
            if not stripped:
                continue
            try:
                values = shlex.split(stripped, comments=False, posix=True)
            except ValueError:
                value_buffer.clear()
                continue
            value_buffer.extend((value, source_line_index) for value in values)
            while len(value_buffer) >= len(headers):
                row_values = value_buffer[: len(headers)]
                del value_buffer[: len(headers)]
                record = _atom_site_record(
                    [value for value, _ in row_values],
                    field_indices,
                    source_line_indices=tuple(dict.fromkeys(index for _, index in row_values)),
                )
                if record is not None:
                    yield record


def filter_mmcif_atom_site_records_by_residue_name(
    structure_text: str,
    *,
    excluded_residue_names: frozenset[str],
) -> str:
    """Remove parsed atom-site rows for excluded residues without corrupting shared lines."""

    excluded_line_indices: set[int] = set()
    retained_line_indices: set[int] = set()
    for record in iter_mmcif_atom_site_records(structure_text):
        target = excluded_line_indices if record.residue_name in excluded_residue_names else retained_line_indices
        target.update(record.source_line_indices)
    removable_line_indices = excluded_line_indices - retained_line_indices
    return "\n".join(
        line for index, line in enumerate(structure_text.splitlines()) if index not in removable_line_indices
    )


def _atom_site_field_indices(headers: list[str]) -> dict[str, tuple[int, ...]] | None:
    header_indices = {header: index for index, header in enumerate(headers)}

    def available(*names: str) -> tuple[int, ...]:
        return tuple(header_indices[name] for name in names if name in header_indices)

    indices = {
        "group": available("_atom_site.group_pdb"),
        "atom": available("_atom_site.label_atom_id", "_atom_site.auth_atom_id"),
        "residue": available("_atom_site.label_comp_id", "_atom_site.auth_comp_id"),
        "chain": available("_atom_site.auth_asym_id", "_atom_site.label_asym_id"),
        "sequence": available("_atom_site.auth_seq_id", "_atom_site.label_seq_id"),
        "insertion": available("_atom_site.pdbx_pdb_ins_code"),
    }
    if any(not indices[field] for field in ("group", "atom", "residue", "chain", "sequence")):
        return None
    return indices


def _atom_site_record(
    row: list[str],
    field_indices: dict[str, tuple[int, ...]],
    *,
    source_line_indices: tuple[int, ...],
) -> MmcifAtomSiteRecord | None:
    group = _first_known_value(row, field_indices["group"])
    atom_name = _first_known_value(row, field_indices["atom"])
    residue_name = _first_known_value(row, field_indices["residue"])
    chain_id = _first_known_value(row, field_indices["chain"])
    residue_number = _first_known_value(row, field_indices["sequence"])
    if group is None or group.upper() not in {"ATOM", "HETATM"}:
        return None
    if None in {atom_name, residue_name, chain_id, residue_number}:
        return None
    return MmcifAtomSiteRecord(
        atom_name=atom_name.upper(),
        residue_name=residue_name.upper(),
        chain_id=chain_id,
        residue_number=residue_number,
        insertion_code=_first_known_value(row, field_indices["insertion"]) or "",
        source_line_indices=source_line_indices,
    )


def _first_known_value(row: list[str], indices: tuple[int, ...]) -> str | None:
    for index in indices:
        value = row[index].strip()
        if value not in {"", ".", "?"}:
            return value
    return None


def _skip_loop_values(lines: list[str], line_index: int) -> int:
    while line_index < len(lines) and not _is_loop_boundary(lines[line_index].strip()):
        line_index += 1
    return line_index


def _is_loop_boundary(stripped_line: str) -> bool:
    lowered = stripped_line.lower()
    return (
        stripped_line == "#"
        or stripped_line.startswith("_")
        or lowered == "loop_"
        or lowered == "stop_"
        or lowered.startswith("data_")
        or lowered.startswith("save_")
    )
