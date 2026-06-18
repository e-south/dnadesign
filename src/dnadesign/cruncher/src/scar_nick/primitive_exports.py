"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/scar_nick/primitive_exports.py

Public primitive export helpers for scar_nick retained base-junction bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dnadesign.cruncher.scar_nick.artifacts import candidate_table_path


class ScarNickPrimitiveExportError(ValueError):
    """Raised when a public scar_nick primitive export cannot be read safely."""


@dataclass(frozen=True)
class ScarNickStemBasePrimitive:
    rank: int
    primitive_id: str
    left_base: str
    right_base: str
    profile_s3s2s1s0: str
    nickase_variant_id: str
    nicked_strand: str
    surviving_strand: str
    source_table: str


def load_scar_nick_stem_base_primitives(run_dir: str | Path) -> list[ScarNickStemBasePrimitive]:
    """Load retained four-base stem-base primitive options from a scar_nick bundle."""

    run_path = Path(run_dir).expanduser().resolve()
    table_path = candidate_table_path(run_path)
    if not table_path.is_file():
        raise ScarNickPrimitiveExportError(f"scar_nick candidate table not found: {table_path}")
    with table_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(_REQUIRED_COLUMNS - fieldnames)
        if missing:
            raise ScarNickPrimitiveExportError(
                f"scar_nick candidate table is missing required columns at {table_path}: {', '.join(missing)}"
            )
        primitives = [
            _primitive_from_row(row, row_number=row_number, table_path=table_path)
            for row_number, row in enumerate(reader, start=2)
        ]
    return sorted(primitives, key=lambda primitive: primitive.rank)


_REQUIRED_COLUMNS = {
    "rank",
    "candidate_id",
    "left_base",
    "right_base",
    "profile_s3s2s1s0",
    "nickase_variant_id",
    "nicked_strand",
    "surviving_strand",
}


def _primitive_from_row(row: dict[str, Any], *, row_number: int, table_path: Path) -> ScarNickStemBasePrimitive:
    rank = _positive_int(row.get("rank"), label=f"row {row_number} rank", table_path=table_path)
    candidate_id = _not_blank(row.get("candidate_id"), label=f"row {row_number} candidate_id")
    return ScarNickStemBasePrimitive(
        rank=rank,
        primitive_id=candidate_id,
        left_base=_dna4(row.get("left_base"), label=f"row {row_number} left_base", table_path=table_path),
        right_base=_dna4(row.get("right_base"), label=f"row {row_number} right_base", table_path=table_path),
        profile_s3s2s1s0=_profile(
            row.get("profile_s3s2s1s0"), label=f"row {row_number} profile_s3s2s1s0", table_path=table_path
        ),
        nickase_variant_id=_not_blank(row.get("nickase_variant_id"), label=f"row {row_number} nickase_variant_id"),
        nicked_strand=str(row.get("nicked_strand") or "").strip(),
        surviving_strand=str(row.get("surviving_strand") or "").strip(),
        source_table=table_path.as_posix(),
    )


def _positive_int(value: Any, *, label: str, table_path: Path) -> int:
    if isinstance(value, bool):
        raise ScarNickPrimitiveExportError(f"scar_nick field {label} must be a positive integer: {table_path}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ScarNickPrimitiveExportError(f"scar_nick field {label} must be a positive integer: {table_path}") from exc
    if parsed < 1:
        raise ScarNickPrimitiveExportError(f"scar_nick field {label} must be >= 1: {table_path}")
    return parsed


def _not_blank(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ScarNickPrimitiveExportError(f"scar_nick field {label} cannot be empty.")
    return text


def _dna4(value: Any, *, label: str, table_path: Path) -> str:
    text = _not_blank(value, label=label).upper()
    if len(text) != 4 or set(text) - {"A", "C", "G", "T"}:
        raise ScarNickPrimitiveExportError(
            f"scar_nick field {label} must contain exactly four A/C/G/T bases: {table_path}"
        )
    return text


def _profile(value: Any, *, label: str, table_path: Path) -> str:
    text = _not_blank(value, label=label).upper()
    if len(text) != 4 or set(text) - {"M", "W", "X"}:
        raise ScarNickPrimitiveExportError(
            f"scar_nick field {label} must contain exactly four M/W/X symbols: {table_path}"
        )
    return text


__all__ = [
    "ScarNickPrimitiveExportError",
    "ScarNickStemBasePrimitive",
    "load_scar_nick_stem_base_primitives",
]
