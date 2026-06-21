"""Exemplar-row contract parsing for generic MSA visualizations."""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.aligner.msa.visualization.contracts.models import (
    ExemplarRow,
    ExemplarRowsSpec,
)


def load_exemplar_rows(path: Path | None) -> ExemplarRowsSpec:
    """Load explicit display-row selections."""

    if path is None:
        return ExemplarRowsSpec(default_rows=(), profile_rows={})
    if not path.exists():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("exemplar rows YAML must be a mapping")
    default_rows = _parse_exemplar_rows(payload.get("rows"), context="default rows", required=False)
    profile_rows = _parse_profile_exemplar_rows(payload.get("profiles"))
    if not default_rows and not profile_rows:
        raise ValueError("exemplar rows YAML must define rows or profiles")
    return ExemplarRowsSpec(default_rows=default_rows, profile_rows=profile_rows)


def validate_exemplar_rows(*, profile_id: str, rows: tuple[ExemplarRow, ...], records: dict[str, str]) -> None:
    """Validate selected display rows against an aligned FASTA."""

    for row in rows:
        if row.record_id not in records:
            raise ValueError(f"{profile_id} exemplar row {row.record_id} is not present in aligned FASTA")


def _parse_profile_exemplar_rows(value: object) -> dict[str, tuple[ExemplarRow, ...]]:
    if value is None:
        return {}
    if not isinstance(value, dict) or not value:
        raise ValueError("exemplar profiles must be a non-empty mapping")
    profile_rows: dict[str, tuple[ExemplarRow, ...]] = {}
    for profile_id, raw_profile in value.items():
        if not isinstance(profile_id, str) or not profile_id.strip():
            raise ValueError("exemplar profile ids must be non-empty strings")
        if not isinstance(raw_profile, dict):
            raise ValueError(f"exemplar profile {profile_id} must be a mapping")
        profile_rows[profile_id] = _parse_exemplar_rows(
            raw_profile.get("rows"),
            context=f"profile {profile_id} rows",
            required=True,
        )
    return profile_rows


def _parse_exemplar_rows(value: object, *, context: str, required: bool) -> tuple[ExemplarRow, ...]:
    raw_rows = value
    if raw_rows is None and not required:
        return ()
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError(f"exemplar rows YAML must define a non-empty {context} list")
    rows: list[ExemplarRow] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            raise ValueError("exemplar row entries must be mappings")
        rows.append(
            ExemplarRow(
                record_id=_required_string(raw_row, "record_id"),
                label=_required_string(raw_row, "label"),
                group=_required_string(raw_row, "group"),
            )
        )
    if len({row.record_id for row in rows}) != len(rows):
        raise ValueError("exemplar row record_id values must be unique")
    return tuple(rows)


def _required_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"exemplar row {key} must be a non-empty string")
    return value
