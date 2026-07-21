"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/synthesis_handoff/azenta.py

Azenta/GeneWiz workbook projection for synthesis manifests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import re
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

AZENTA_SHEET_NAME = "Azenta Gene Synthesis"
AZENTA_COLUMNS: tuple[str, ...] = (
    "Sequence Name",
    "Sequence",
    "Add Protection Nt.",
    "5' Phosphorylation",
)
_MANIFEST_COLUMNS = ("synthesis_name", "final_sequence")
_WORKBOOK_TIMESTAMP = datetime(2000, 1, 1)
_ZIP_TIMESTAMP = (2000, 1, 1, 0, 0, 0)
_CORE_MODIFIED_PATTERN = re.compile(rb"(<dcterms:modified[^>]*>)[^<]*(</dcterms:modified>)")


def _require_manifest_columns(manifest: pd.DataFrame) -> None:
    missing = [column for column in _MANIFEST_COLUMNS if column not in manifest.columns]
    if missing:
        raise ValueError("synthesis manifest missing required columns: " + ", ".join(missing))


def azenta_rows_from_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Project a vendor-neutral manifest into Azenta/GeneWiz order rows."""

    _require_manifest_columns(manifest)
    rows = pd.DataFrame(
        {
            "Sequence Name": manifest["synthesis_name"].astype(str),
            "Sequence": manifest["final_sequence"].astype(str),
            "Add Protection Nt.": "",
            "5' Phosphorylation": "",
        }
    )
    return rows.loc[:, list(AZENTA_COLUMNS)]


def render_azenta_workbook(manifest: pd.DataFrame, path: str | Path) -> Path:
    """Write an Azenta/GeneWiz workbook projection for a manifest."""

    workbook_path = Path(path)
    workbook_path.parent.mkdir(parents=True, exist_ok=True)
    rows = azenta_rows_from_manifest(manifest)
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        writer.book.properties.created = _WORKBOOK_TIMESTAMP
        writer.book.properties.modified = _WORKBOOK_TIMESTAMP
        rows.to_excel(writer, sheet_name=AZENTA_SHEET_NAME, index=False)
    _normalize_xlsx_archive(workbook_path)
    return workbook_path


def _normalize_xlsx_archive(path: Path) -> None:
    """Rewrite XLSX member metadata so identical rows have identical bytes."""

    with zipfile.ZipFile(path, "r") as source:
        members = [(info, source.read(info.filename)) for info in source.infolist()]

    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with zipfile.ZipFile(temp_path, "w") as target:
            for source_info, content in members:
                if source_info.filename == "docProps/core.xml":
                    content = _CORE_MODIFIED_PATTERN.sub(
                        rb"\g<1>2000-01-01T00:00:00Z\g<2>",
                        content,
                    )
                info = zipfile.ZipInfo(filename=source_info.filename, date_time=_ZIP_TIMESTAMP)
                info.compress_type = source_info.compress_type
                info.comment = source_info.comment
                info.extra = source_info.extra
                info.internal_attr = source_info.internal_attr
                info.external_attr = source_info.external_attr
                info.create_system = source_info.create_system
                target.writestr(info, content)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def read_azenta_workbook(path: str | Path) -> pd.DataFrame:
    """Read the Azenta/GeneWiz workbook sheet used by this handoff."""

    workbook_path = Path(path)
    if not workbook_path.exists():
        raise ValueError(f"Azenta workbook not found: {workbook_path}")
    try:
        rows = pd.read_excel(workbook_path, sheet_name=AZENTA_SHEET_NAME, dtype=str).fillna("")
    except ValueError as exc:
        raise ValueError(f"Azenta workbook missing sheet {AZENTA_SHEET_NAME!r}: {workbook_path}") from exc
    observed_columns = tuple(str(column) for column in rows.columns)
    if observed_columns != AZENTA_COLUMNS:
        raise ValueError(
            "Azenta workbook column contract mismatch: "
            f"expected {list(AZENTA_COLUMNS)!r}, observed {list(observed_columns)!r}"
        )
    return rows


def validate_azenta_workbook(manifest: pd.DataFrame, path: str | Path) -> dict[str, Any]:
    """Validate workbook aliases and sequences against the canonical manifest."""

    _require_manifest_columns(manifest)
    workbook_path = Path(path)
    expected = azenta_rows_from_manifest(manifest).astype(str).reset_index(drop=True)
    observed = read_azenta_workbook(workbook_path).astype(str).reset_index(drop=True)

    if len(observed) != len(expected):
        raise ValueError(f"Azenta workbook row count mismatch: expected {len(expected)}, observed {len(observed)}")

    alias_mismatches = observed["Sequence Name"] != expected["Sequence Name"]
    if bool(alias_mismatches.any()):
        first = int(alias_mismatches[alias_mismatches].index[0])
        raise ValueError(
            "Azenta workbook synthesis_name mismatch at row "
            f"{first + 2}: expected {expected.loc[first, 'Sequence Name']!r}, "
            f"observed {observed.loc[first, 'Sequence Name']!r}"
        )

    sequence_mismatches = observed["Sequence"] != expected["Sequence"]
    if bool(sequence_mismatches.any()):
        first = int(sequence_mismatches[sequence_mismatches].index[0])
        raise ValueError(
            "Azenta workbook sequence mismatch at row "
            f"{first + 2}: expected manifest final_sequence for {expected.loc[first, 'Sequence Name']!r}"
        )

    for column in ("Add Protection Nt.", "5' Phosphorylation"):
        option_mismatches = observed[column] != expected[column]
        if bool(option_mismatches.any()):
            first = int(option_mismatches[option_mismatches].index[0])
            raise ValueError(
                f"Azenta workbook {column} mismatch at row {first + 2}: "
                f"expected {expected.loc[first, column]!r}, observed {observed.loc[first, column]!r}"
            )

    return {"status": "pass", "row_count": int(len(expected)), "workbook_path": str(workbook_path)}
