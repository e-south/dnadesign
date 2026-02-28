"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/pipeline/selection.py

Strict CSV selection service for matching, overlay labeling, and missing-key policy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import Mapping, Sequence

from ..config import SelectionCfg
from ..core import Record, SchemaError
from ..core.record import Display


@dataclass(frozen=True)
class SelectionRows:
    keys: list[str]
    overlays: list[str | None]
    overlay_by_key: dict[str, str]


def read_selection_rows(path, *, key_col: str, overlay_col: str | None) -> SelectionRows:
    keys: list[str] = []
    overlays: list[str | None] = []
    overlay_by_key: dict[str, str] = {}

    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or key_col not in reader.fieldnames:
            raise SchemaError(f"Selection CSV must contain key column '{key_col}'")
        if overlay_col is not None and overlay_col not in reader.fieldnames:
            raise SchemaError(f"Selection CSV is missing required overlay column '{overlay_col}'")

        for row in reader:
            raw_key = row.get(key_col)
            if raw_key is None:
                continue
            key = str(raw_key).strip()
            if key == "":
                continue
            keys.append(key)

            overlay: str | None = None
            if overlay_col is not None:
                raw_overlay = row.get(overlay_col)
                if raw_overlay is not None:
                    sval = str(raw_overlay).strip()
                    if sval != "":
                        overlay = sval
                        overlay_by_key[key] = sval
            overlays.append(overlay)

    if not keys:
        raise SchemaError(f"Selection CSV column '{key_col}' contains no non-blank values")
    return SelectionRows(keys=keys, overlays=overlays, overlay_by_key=overlay_by_key)


def _with_overlay(record: Record, text: str | None) -> Record:
    if text is None or text.strip() == "":
        return record
    return Record(
        id=record.id,
        alphabet=record.alphabet,
        sequence=record.sequence,
        features=record.features,
        effects=record.effects,
        display=Display(
            overlay_text=text,
            tag_labels=dict(record.display.tag_labels),
            trajectory_inset=record.display.trajectory_inset,
        ),
        meta=dict(record.meta),
    ).validate()


def _record_key_map(records: Sequence[Record], *, match_on: str) -> Mapping[str, Record]:
    mapping: dict[str, Record] = {}
    duplicates: set[str] = set()

    def _bind(key: str, record: Record) -> None:
        if key in mapping:
            duplicates.add(key)
            return
        mapping[key] = record

    if match_on == "id":
        for record in records:
            _bind(record.id, record)
    elif match_on == "sequence":
        for record in records:
            _bind(record.sequence, record)
    else:
        for record in records:
            idx = record.meta.get("row_index")
            if idx is None:
                continue
            try:
                row_index = int(idx)
            except Exception as exc:
                raise SchemaError("record.meta.row_index must be int for selection.match_on=row") from exc
            _bind(str(row_index), record)

    if duplicates:
        sample = sorted(duplicates)[:5]
        raise SchemaError(f"selection.match_on={match_on} produced duplicate record keys; examples: {sample}")

    return mapping


def _ordered_keys(selection: SelectionCfg, rows: SelectionRows, mapping: Mapping[str, Record]) -> list[str]:
    if selection.keep_order:
        return list(rows.keys)

    present = {key for key in rows.keys if key in mapping}
    if selection.match_on == "row":
        return sorted(present, key=lambda k: int(k))
    return sorted(present)


def _default_overlay(selection: SelectionCfg, *, sel_row: int, key: str, record: Record) -> str:
    if selection.match_on == "row":
        return f"sel_row={sel_row} row={key} id={record.id}"
    return f"sel_row={sel_row} id={record.id}"


def apply_selection(records: Sequence[Record], selection: SelectionCfg) -> tuple[list[Record], list[str]]:
    rows = read_selection_rows(selection.path, key_col=selection.column, overlay_col=selection.overlay_column)
    mapping = _record_key_map(records, match_on=selection.match_on)

    missing = [key for key in rows.keys if key not in mapping]
    ordered_keys = _ordered_keys(selection, rows, mapping)

    selected: list[Record] = []
    for i, key in enumerate(ordered_keys):
        rec = mapping.get(key)
        if rec is None:
            continue

        if selection.keep_order and i < len(rows.overlays):
            overlay = rows.overlays[i]
        else:
            overlay = rows.overlay_by_key.get(key)

        if overlay is None and rec.display.overlay_text is None:
            overlay = _default_overlay(selection, sel_row=i, key=key, record=rec)
        selected.append(_with_overlay(rec, overlay))

    return selected, missing


def enforce_selection_policy(selection: SelectionCfg, missing: Sequence[str]) -> None:
    if not missing:
        return
    if selection.on_missing == "error":
        sample = list(missing)[:5]
        raise SchemaError(f"{len(missing)} selection keys were not found. Examples: {sample}")
