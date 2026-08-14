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
from io import StringIO
from typing import Iterator

from Bio.PDB.MMCIF2Dict import MMCIF2Dict

_BROWSER_ATOM_SITE_FIELDS: tuple[str, ...] = (
    "_atom_site.group_pdb",
    "_atom_site.id",
    "_atom_site.type_symbol",
    "_atom_site.label_atom_id",
    "_atom_site.label_alt_id",
    "_atom_site.label_comp_id",
    "_atom_site.label_asym_id",
    "_atom_site.label_seq_id",
    "_atom_site.cartn_x",
    "_atom_site.cartn_y",
    "_atom_site.cartn_z",
    "_atom_site.auth_asym_id",
    "_atom_site.auth_seq_id",
    "_atom_site.pdbx_pdb_ins_code",
    "_atom_site.occupancy",
    "_atom_site.b_iso_or_equiv",
    "_atom_site.pdbx_pdb_model_num",
)


@dataclass(frozen=True)
class MmcifAtomSiteRecord:
    """Identity fields and source lines for one parsed atom-site record."""

    atom_name: str
    residue_name: str
    chain_id: str
    residue_number: str
    insertion_code: str
    source_line_indices: tuple[int, ...]


def serialize_mmcif_atom_sites_for_3dmol(structure_text: str) -> str:
    """Return a coordinate-only mmCIF payload safe for 3Dmol's CIF tokenizer.

    3Dmol 2.5.5 treats apostrophes inside otherwise valid unquoted CIF tokens as
    quote delimiters. Nucleic-acid atom names such as ``O5'`` then shift the
    remaining atom-site columns. This adapter emits one canonical coordinate
    loop and double-quotes every atom name, which 3Dmol explicitly unquotes.
    """

    raw = MMCIF2Dict(StringIO(structure_text))
    source = {str(key).lower(): _as_mmcif_values(value) for key, value in raw.items()}
    fields = _browser_atom_site_values(source)
    row_count = len(fields["_atom_site.group_pdb"])
    if row_count == 0:
        raise ValueError("mmCIF browser serialization requires at least one atom-site record")
    inconsistent = {field: len(values) for field, values in fields.items() if len(values) != row_count}
    if inconsistent:
        raise ValueError(
            "mmCIF browser serialization received inconsistent atom-site column lengths: "
            f"expected {row_count}, got {inconsistent}"
        )

    lines = ["data_dnadesign_browser", "loop_", *_BROWSER_ATOM_SITE_FIELDS]
    for row_index in range(row_count):
        row = [fields[field][row_index] for field in _BROWSER_ATOM_SITE_FIELDS]
        row[3] = _quote_3dmol_atom_name(row[3])
        lines.append(
            " ".join(
                _browser_cif_token(value, field=_BROWSER_ATOM_SITE_FIELDS[index]) for index, value in enumerate(row)
            )
        )
    lines.append("#")
    return "\n".join(lines)


def _as_mmcif_values(value: object) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    return (str(value),)


def _browser_atom_site_values(source: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    group = _required_atom_site_column(source, "_atom_site.group_pdb")
    row_count = len(group)

    def column(
        name: str,
        *fallback_names: str,
        default: str | None = None,
    ) -> tuple[str, ...]:
        for candidate in (name, *fallback_names):
            values = source.get(candidate)
            if values is not None:
                return values
        if default is not None:
            return (default,) * row_count
        joined = ", ".join((name, *fallback_names))
        raise ValueError(f"mmCIF browser serialization requires one of these atom-site columns: {joined}")

    return {
        "_atom_site.group_pdb": group,
        "_atom_site.id": source.get("_atom_site.id", tuple(str(index + 1) for index in range(row_count))),
        "_atom_site.type_symbol": column("_atom_site.type_symbol"),
        "_atom_site.label_atom_id": column("_atom_site.label_atom_id", "_atom_site.auth_atom_id"),
        "_atom_site.label_alt_id": column("_atom_site.label_alt_id", default="."),
        "_atom_site.label_comp_id": column("_atom_site.label_comp_id", "_atom_site.auth_comp_id"),
        "_atom_site.label_asym_id": column("_atom_site.label_asym_id", "_atom_site.auth_asym_id"),
        "_atom_site.label_seq_id": column("_atom_site.label_seq_id", "_atom_site.auth_seq_id"),
        "_atom_site.cartn_x": column("_atom_site.cartn_x"),
        "_atom_site.cartn_y": column("_atom_site.cartn_y"),
        "_atom_site.cartn_z": column("_atom_site.cartn_z"),
        "_atom_site.auth_asym_id": column("_atom_site.auth_asym_id", "_atom_site.label_asym_id"),
        "_atom_site.auth_seq_id": column("_atom_site.auth_seq_id", "_atom_site.label_seq_id"),
        "_atom_site.pdbx_pdb_ins_code": column("_atom_site.pdbx_pdb_ins_code", default="?"),
        "_atom_site.occupancy": column("_atom_site.occupancy", default="1.00"),
        "_atom_site.b_iso_or_equiv": column("_atom_site.b_iso_or_equiv", default="0.00"),
        "_atom_site.pdbx_pdb_model_num": column("_atom_site.pdbx_pdb_model_num", default="1"),
    }


def _required_atom_site_column(
    source: dict[str, tuple[str, ...]],
    name: str,
) -> tuple[str, ...]:
    values = source.get(name)
    if values is None:
        raise ValueError(f"mmCIF browser serialization requires atom-site column {name}")
    return values


def _quote_3dmol_atom_name(value: str) -> str:
    if not value or any(character in value for character in ('"', "\n", "\r")):
        raise ValueError(f"mmCIF atom name cannot be serialized safely for 3Dmol: {value!r}")
    return f'"{value}"'


def _browser_cif_token(value: str, *, field: str) -> str:
    if field == "_atom_site.label_atom_id" and value.startswith('"') and value.endswith('"'):
        return value
    if (
        not value
        or any(character.isspace() for character in value)
        or any(character in value for character in ("'", '"'))
    ):
        raise ValueError(f"mmCIF value cannot be serialized safely for 3Dmol at {field}: {value!r}")
    return value


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
