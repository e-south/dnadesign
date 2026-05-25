"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/api/codon_tables.py

Public codon-table helpers.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal


def default_codon_table_path(table_id: Literal["ecoli"] = "ecoli") -> Path:
    if table_id != "ecoli":
        raise ValueError(f"Unsupported default codon table: {table_id!r}")
    root = Path(__file__).resolve().parents[2]
    path = root / "src" / "resources" / "codon_tables" / "codon_ecoli.csv"
    if not path.exists():
        raise ValueError(f"Default E. coli codon table is not available: {path}")
    return path
