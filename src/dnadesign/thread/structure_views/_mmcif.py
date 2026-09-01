"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/structure_views/_mmcif.py

Fail-safe helpers for reading mmCIF coordinate loops.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import Iterator, TextIO

from Bio.PDB.MMCIF2Dict import MMCIF2Dict

_CIF_QUOTE_CHARS = frozenset({"'", '"'})
_CIF_WHITESPACE_CHARS = frozenset({" ", "\t"})
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


@dataclass(frozen=True)
class _BrowserAtomSiteColumn:
    values: tuple[str, ...]
    source_quoted: tuple[bool | None, ...]


class _CifSourceToken(str):
    """String token carrying whether CIF syntax made it explicit literal text."""

    quoted: bool

    def __new__(cls, value: str, *, quoted: bool) -> _CifSourceToken:
        token = super().__new__(cls, value)
        token.quoted = quoted
        return token


def _split_cif_line(line: str) -> Iterator[_CifSourceToken]:
    """Yield source-aware CIF tokens from one non-semicolon data line."""

    in_token = False
    quote_open: str | None = None
    start_index = 0
    for index, character in enumerate(line):
        if character in _CIF_WHITESPACE_CHARS:
            if in_token and quote_open is None:
                in_token = False
                yield _CifSourceToken(line[start_index:index], quoted=False)
        elif character in _CIF_QUOTE_CHARS:
            if quote_open is None and not in_token:
                quote_open = character
                in_token = True
                start_index = index + 1
            elif character == quote_open and (index + 1 == len(line) or line[index + 1] in _CIF_WHITESPACE_CHARS):
                quote_open = None
                in_token = False
                yield _CifSourceToken(line[start_index:index], quoted=True)
        elif character == "#" and not in_token:
            return
        elif not in_token:
            in_token = True
            start_index = index
    if in_token:
        yield _CifSourceToken(line[start_index:], quoted=False)
    if quote_open is not None:
        raise ValueError(f"mmCIF line ended with an open quote: {line}")


class _SourceAwareMMCIF2Dict(MMCIF2Dict):
    """Retain literal-vs-null provenance discarded from public dictionary values."""

    def _tokenize(self, handle: TextIO) -> Iterator[str]:
        for token in super()._tokenize(handle):
            if isinstance(token, _CifSourceToken):
                yield token
            else:
                yield _CifSourceToken(token, quoted=True)

    def _splitline(self, line: str) -> Iterator[str]:
        yield from _split_cif_line(line)


def serialize_mmcif_atom_sites_for_3dmol(
    structure_text: str,
    *,
    allow_filtered_empty: bool = False,
) -> str:
    """Return a coordinate-only mmCIF payload safe for 3Dmol's CIF tokenizer.

    3Dmol 2.5.5 treats apostrophes inside otherwise valid unquoted CIF tokens as
    quote delimiters. Nucleic-acid atom names such as ``O5'`` then shift the
    remaining atom-site columns. This adapter emits one canonical coordinate
    loop and quotes every atom name with an available delimiter that 3Dmol
    explicitly unquotes. It also retains source quote provenance so unquoted
    CIF null markers remain null while quoted ``'.'`` and ``'?'`` values remain
    literal strings. ``allow_filtered_empty`` is reserved for the renderer's
    proven nonempty-source-to-empty-filter transition; native empty inputs remain
    invalid by default.
    """

    raw = _SourceAwareMMCIF2Dict(StringIO(structure_text))
    source = {str(key).lower(): _as_mmcif_values(value) for key, value in raw.items()}
    source_quote_statuses = {str(key).lower(): _as_mmcif_quote_statuses(value) for key, value in raw.items()}
    columns = _browser_atom_site_columns(source, source_quote_statuses=source_quote_statuses)
    row_count = len(columns["_atom_site.group_pdb"].values)
    if row_count == 0 and not allow_filtered_empty:
        raise ValueError("mmCIF browser serialization requires at least one atom-site record")
    inconsistent = {field: len(column.values) for field, column in columns.items() if len(column.values) != row_count}
    if inconsistent:
        raise ValueError(
            "mmCIF browser serialization received inconsistent atom-site column lengths: "
            f"expected {row_count}, got {inconsistent}"
        )

    lines = ["data_dnadesign_browser", "loop_", *_BROWSER_ATOM_SITE_FIELDS]
    for row_index in range(row_count):
        row = [columns[field].values[row_index] for field in _BROWSER_ATOM_SITE_FIELDS]
        source_quoted = [columns[field].source_quoted[row_index] for field in _BROWSER_ATOM_SITE_FIELDS]
        row[3] = _quote_3dmol_atom_name(row[3])
        lines.append(
            " ".join(
                _browser_cif_token(
                    value,
                    field=_BROWSER_ATOM_SITE_FIELDS[index],
                    source_quoted=source_quoted[index],
                )
                for index, value in enumerate(row)
            )
        )
    lines.append("#")
    return "\n".join(lines)


def _as_mmcif_values(value: object) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    return (str(value),)


def _as_mmcif_quote_statuses(value: object) -> tuple[bool | None, ...]:
    items = value if isinstance(value, list) else [value]
    return tuple(item.quoted if isinstance(item, _CifSourceToken) else None for item in items)


def _browser_atom_site_columns(
    source: dict[str, tuple[str, ...]],
    *,
    source_quote_statuses: dict[str, tuple[bool | None, ...]],
) -> dict[str, _BrowserAtomSiteColumn]:
    group = _required_atom_site_column(source, "_atom_site.group_pdb")
    row_count = len(group)

    def column(
        name: str,
        *fallback_names: str,
        default: str | None = None,
    ) -> _BrowserAtomSiteColumn:
        for candidate in (name, *fallback_names):
            values = source.get(candidate)
            if values is not None:
                quoted = source_quote_statuses.get(candidate, (None,) * len(values))
                if len(quoted) != len(values):
                    raise ValueError(f"mmCIF source quote provenance does not align for {candidate}")
                return _BrowserAtomSiteColumn(values=values, source_quoted=quoted)
        if default is not None:
            return _BrowserAtomSiteColumn(
                values=(default,) * row_count,
                source_quoted=(False,) * row_count,
            )
        joined = ", ".join((name, *fallback_names))
        raise ValueError(f"mmCIF browser serialization requires one of these atom-site columns: {joined}")

    def required_concrete_column(name: str, fallback_name: str) -> _BrowserAtomSiteColumn:
        candidates = [
            (candidate_name, column(candidate_name))
            for candidate_name in (name, fallback_name)
            if candidate_name in source
        ]
        if not candidates:
            raise ValueError(
                f"mmCIF browser serialization requires one of these atom-site columns: {name}, {fallback_name}"
            )
        for candidate_name, candidate in candidates:
            if len(candidate.values) != row_count:
                raise ValueError(
                    "mmCIF browser serialization received inconsistent atom-site column lengths: "
                    f"expected {row_count}, got {candidate_name}={len(candidate.values)}"
                )

        values: list[str] = []
        quoted: list[bool | None] = []
        for row_index in range(row_count):
            for candidate_name, candidate in candidates:
                value = candidate.values[row_index]
                source_quoted = candidate.source_quoted[row_index]
                if not _is_unquoted_cif_null(value, source_quoted=source_quoted, field=candidate_name):
                    values.append(value)
                    quoted.append(source_quoted)
                    break
            else:
                raise ValueError(
                    "mmCIF browser serialization requires a concrete label_atom_id or auth_atom_id "
                    f"at row {row_index + 1}"
                )
        return _BrowserAtomSiteColumn(values=tuple(values), source_quoted=tuple(quoted))

    return {
        "_atom_site.group_pdb": column("_atom_site.group_pdb"),
        "_atom_site.id": (
            column("_atom_site.id")
            if "_atom_site.id" in source
            else _BrowserAtomSiteColumn(
                values=tuple(str(index + 1) for index in range(row_count)),
                source_quoted=(False,) * row_count,
            )
        ),
        "_atom_site.type_symbol": column("_atom_site.type_symbol"),
        "_atom_site.label_atom_id": required_concrete_column(
            "_atom_site.label_atom_id",
            "_atom_site.auth_atom_id",
        ),
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
    if not value or any(character in value for character in ("\n", "\r")):
        raise ValueError(f"mmCIF atom name cannot be serialized safely for 3Dmol: {value!r}")
    if '"' not in value:
        return f'"{value}"'
    if "'" not in value:
        return f"'{value}'"
    raise ValueError(f"mmCIF atom name cannot be serialized safely for 3Dmol: {value!r}")


def _browser_cif_token(value: str, *, field: str, source_quoted: bool | None = None) -> str:
    if field == "_atom_site.label_atom_id" and len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value
    if not value or any(character in value for character in ("\n", "\r")):
        raise ValueError(f"mmCIF value cannot be serialized safely for 3Dmol at {field}: {value!r}")
    if _is_unquoted_cif_null(value, source_quoted=source_quoted, field=field):
        return value
    lowered = value.casefold()
    requires_quotes = (
        any(character.isspace() for character in value)
        or any(character in value for character in ("'", '"'))
        or value in {".", "?"}
        or value.startswith(("_", "#", "$", "[", "]"))
        or lowered.startswith(("data_", "save_"))
        or lowered in {"loop_", "stop_", "global_"}
    )
    if requires_quotes:
        if '"' not in value:
            return f'"{value}"'
        if "'" not in value:
            return f"'{value}'"
        raise ValueError(f"mmCIF value cannot be serialized safely for 3Dmol at {field}: {value!r}")
    return value


def _is_unquoted_cif_null(value: str, *, source_quoted: bool | None, field: str) -> bool:
    if value not in {".", "?"}:
        return False
    if source_quoted is None:
        raise ValueError(f"mmCIF null-marker quote provenance is unknown at {field}")
    return not source_quoted


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

        value_buffer: list[tuple[str, tuple[int, ...]]] = []
        while line_index < len(lines):
            stripped = lines[line_index].strip()
            if _is_loop_boundary(stripped):
                break
            source_line_index = line_index
            if not stripped:
                line_index += 1
                continue
            if lines[source_line_index].startswith(";"):
                value, source_line_indices, line_index = _consume_semicolon_cif_token(
                    lines,
                    source_line_index,
                )
                values = (value,)
                value_source_lines = (source_line_indices,)
            else:
                line_index += 1
                try:
                    values = tuple(_split_cif_line(stripped))
                except ValueError:
                    value_buffer.clear()
                    continue
                value_source_lines = ((source_line_index,),) * len(values)
            value_buffer.extend(zip(values, value_source_lines, strict=True))
            while len(value_buffer) >= len(headers):
                row_values = value_buffer[: len(headers)]
                del value_buffer[: len(headers)]
                record = _atom_site_record(
                    [value for value, _ in row_values],
                    field_indices,
                    source_line_indices=tuple(
                        dict.fromkeys(index for _, source_line_indices in row_values for index in source_line_indices)
                    ),
                )
                if record is not None:
                    yield record


def _consume_semicolon_cif_token(
    lines: list[str],
    opening_line_index: int,
) -> tuple[str, tuple[int, ...], int]:
    """Consume one CIF text-field token and retain every source line it spans."""

    token_lines = [lines[opening_line_index][1:].rstrip()]
    source_line_indices = [opening_line_index]
    line_index = opening_line_index + 1
    while line_index < len(lines):
        line = lines[line_index]
        source_line_indices.append(line_index)
        line_index += 1
        if line.startswith(";"):
            trailing = line[1:]
            if trailing.strip():
                raise ValueError("mmCIF text-field closing delimiter must occupy its own line")
            return "\n".join(token_lines), tuple(source_line_indices), line_index
        token_lines.append(line.rstrip())
    raise ValueError("mmCIF text field is missing its closing semicolon")


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
