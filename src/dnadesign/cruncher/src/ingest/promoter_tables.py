"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/ingest/promoter_tables.py

Small table-normalization helpers shared by promoter ingest modules.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from .promoter_payloads import _text_or_none


def _normalized_table_key(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return re.sub(r"^\d+_", "", normalized)


def _normalized_table_row(row: Mapping[str, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in row.items():
        norm = _normalized_table_key(str(key))
        if norm:
            normalized[norm] = "" if value is None else str(value)
    return normalized


def _table_value(row: Mapping[str, str], *aliases: str) -> str | None:
    for alias in aliases:
        value = _text_or_none(row.get(_normalized_table_key(alias)))
        if value is not None:
            return value
    return None


def _split_table_list(value: str | None) -> tuple[str, ...]:
    text = _text_or_none(value)
    if text is None:
        return ()
    return tuple(part.strip() for part in re.split(r"[;,|]", text) if part.strip())


def _missing_table_value(value: str | None) -> bool:
    text = str(value or "").strip()
    return not text or text.casefold() in {"none", "null", "nan", "na"}


def _iter_delimited_data_rows(path: Path, *, delimiter: str) -> Iterable[tuple[int, dict[str, str]]]:
    header: list[str] | None = None
    with path.open(encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.lstrip('"').startswith("#"):
                continue
            values = next(csv.reader([line], delimiter=delimiter))
            if not values or str(values[0]).strip().startswith("#"):
                continue
            if header is None:
                header = values
                continue
            if len(values) < len(header):
                values = [*values, *([""] * (len(header) - len(values)))]
            yield line_number, _normalized_table_row(dict(zip(header, values, strict=False)))


def _iter_tsv_data_rows(path: Path) -> Iterable[tuple[int, dict[str, str]]]:
    return _iter_delimited_data_rows(path, delimiter="\t")
