"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/selection.py

Selection CSV parsing and overlay label utilities.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Optional

from .contracts import SchemaError
from .model import Guide, SeqRecord


@dataclass(frozen=True)
class SelectionSpec:
    keys: list[str]
    overlays: list[Optional[str]]
    overlay_by_key: dict[str, str]
    key_col: str
    overlay_col: Optional[str]


def read_selection_csv(path: Path, *, key_col: str, overlay_col: Optional[str]) -> SelectionSpec:
    keys: list[str] = []
    overlays: list[Optional[str]] = []
    overlay_by_key: dict[str, str] = {}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or key_col not in reader.fieldnames:
            raise SchemaError(f"CSV '{path}' must contain column '{key_col}' (found: {reader.fieldnames or []})")
        if overlay_col is not None and overlay_col not in reader.fieldnames:
            raise SchemaError(f"CSV '{path}' is missing overlay column '{overlay_col}' (found: {reader.fieldnames})")
        for row in reader:
            raw_key = row.get(key_col)
            if raw_key is None:
                continue
            key = str(raw_key).strip()
            if key == "":
                continue
            keys.append(key)
            text: Optional[str] = None
            if overlay_col is not None:
                raw_text = row.get(overlay_col)
                if raw_text is not None:
                    s = str(raw_text).strip()
                    if s != "":
                        text = s
                        overlay_by_key[key] = s
            overlays.append(text)

    if not keys:
        raise SchemaError(f"CSV '{path}' column '{key_col}' contains no non-blank values.")

    return SelectionSpec(
        keys=keys,
        overlays=overlays,
        overlay_by_key=overlay_by_key,
        key_col=key_col,
        overlay_col=overlay_col,
    )


def _strip_overlay_guides(record: SeqRecord) -> SeqRecord:
    guides = [g for g in record.guides if getattr(g, "kind", "") != "overlay_label"]
    return dc_replace(record, guides=tuple(guides)).validate()


def apply_overlay_label(
    record: SeqRecord,
    label_text: Optional[str],
    *,
    source: str,
) -> SeqRecord:
    if source == "csv":
        if label_text is None or str(label_text).strip() == "":
            return record
        rec = _strip_overlay_guides(record)
        return rec.with_extra(guides=[Guide(kind="overlay_label", start=0, end=0, label=str(label_text))])
    if source == "default":
        if label_text is None:
            return record
        return record.with_extra(guides=[Guide(kind="overlay_label", start=0, end=0, label=str(label_text))])
    if source == "dataset":
        return record
    raise SchemaError(f"Unknown overlay label source: {source!r}")
