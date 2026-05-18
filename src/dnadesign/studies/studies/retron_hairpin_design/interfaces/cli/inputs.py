"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/studies/retron_hairpin_design/interfaces/cli/inputs.py

Retron MSD CLI input parsing helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
from pathlib import Path

from ...compiler.exceptions import RetronMsdCompilerError


def collect_labels(ids: list[str], input_file: Path | None) -> list[str]:
    labels = [item.strip() for item in ids if item.strip()]
    if input_file is not None:
        labels.extend(read_input_labels(input_file))
    if not labels:
        raise RetronMsdCompilerError("Provide at least one --id or an --input file with construct labels.")
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise RetronMsdCompilerError(f"Duplicate construct label(s): {', '.join(duplicates)}")
    return labels


def read_input_labels(input_file: Path) -> list[str]:
    path = input_file.expanduser().resolve()
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".csv", ".tsv", ".tab"}:
        delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
        rows = csv.DictReader(text.splitlines(), delimiter=delimiter)
        if rows.fieldnames is None:
            raise RetronMsdCompilerError(f"Input file has no header row: {path}")
        for field in ("construct_label", "design_id", "id"):
            if field in rows.fieldnames:
                return [str(row.get(field, "")).strip() for row in rows if str(row.get(field, "")).strip()]
        raise RetronMsdCompilerError("CSV/TSV input must include construct_label, design_id, or id column.")
    return [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def reject_mixed_design_sources(*, ids: list[str], input_file: Path | None, spec_file: Path | None) -> None:
    if spec_file is not None and (ids or input_file is not None):
        raise RetronMsdCompilerError("Use either --spec or --id/--input, not both.")


def merge_sequence_maps(
    spec_values: dict[str, str],
    cli_values: dict[str, str],
    *,
    label: str,
) -> dict[str, str]:
    duplicates = sorted(set(spec_values) & set(cli_values))
    if duplicates:
        raise RetronMsdCompilerError(
            f"{label} sequence(s) declared in both --spec and CLI overrides: {', '.join(duplicates)}"
        )
    return {**spec_values, **cli_values}


def sequence_override_map(values: list[str], *, label: str) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for raw in values:
        text = str(raw or "").strip()
        if not text:
            continue
        key, separator, value = text.partition("=")
        if separator != "=" or not key.strip() or not value.strip():
            raise RetronMsdCompilerError(f"{label} override must be ID=SEQUENCE.")
        overrides[key.strip()] = value.strip()
    return overrides


__all__ = [
    "collect_labels",
    "merge_sequence_maps",
    "read_input_labels",
    "reject_mixed_design_sources",
    "sequence_override_map",
]
