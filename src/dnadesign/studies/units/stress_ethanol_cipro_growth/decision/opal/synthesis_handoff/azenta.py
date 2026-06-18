"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/synthesis_handoff/azenta.py

Azenta/GeneWiz workbook projection for synthesis manifests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

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
        rows.to_excel(writer, sheet_name=AZENTA_SHEET_NAME, index=False)
    return workbook_path


def read_azenta_workbook(path: str | Path) -> pd.DataFrame:
    """Read the Azenta/GeneWiz workbook sheet used by this handoff."""

    workbook_path = Path(path)
    if not workbook_path.exists():
        raise ValueError(f"Azenta workbook not found: {workbook_path}")
    try:
        rows = pd.read_excel(workbook_path, sheet_name=AZENTA_SHEET_NAME, dtype=str).fillna("")
    except ValueError as exc:
        raise ValueError(f"Azenta workbook missing sheet {AZENTA_SHEET_NAME!r}: {workbook_path}") from exc
    missing = [column for column in ("Sequence Name", "Sequence") if column not in rows.columns]
    if missing:
        raise ValueError("Azenta workbook missing required columns: " + ", ".join(missing))
    return rows


def validate_azenta_workbook(manifest: pd.DataFrame, path: str | Path) -> dict[str, Any]:
    """Validate workbook aliases and sequences against the canonical manifest."""

    _require_manifest_columns(manifest)
    workbook_path = Path(path)
    expected = (
        manifest.loc[:, list(_MANIFEST_COLUMNS)]
        .rename(columns={"synthesis_name": "Sequence Name", "final_sequence": "Sequence"})
        .astype(str)
        .reset_index(drop=True)
    )
    observed = read_azenta_workbook(workbook_path).loc[:, ["Sequence Name", "Sequence"]]
    observed = observed.astype(str).reset_index(drop=True)

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

    return {"status": "pass", "row_count": int(len(expected)), "workbook_path": str(workbook_path)}
