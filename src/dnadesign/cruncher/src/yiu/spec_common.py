"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/spec_common.py

Shared validation helpers for payload-centric YIU spec models.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.bio import normalize_iupac
from dnadesign.cruncher.yiu.errors import YIU_PATH_INVALID, YIU_SEQUENCE_INVALID

BASES = ("A", "C", "G", "T")
SECONDARY_OBJECTIVE_LADDER = (
    "total_loss",
    "midpoint_proximity",
    "body_length_balance",
    "terminal_position_avoidance",
    "default_strand_preference",
    "lexical_stability",
)


def normalize_yiu_sequence(value: str, *, ctx: str) -> str:
    try:
        return normalize_iupac(value)
    except Exception as exc:
        raise ValueError(f"{YIU_SEQUENCE_INVALID}: invalid {ctx} ({exc})") from exc


def require_non_empty_text(value: str, *, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    return text


def normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def validate_workspace_relative_path(*, value: Path, field_name: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{YIU_PATH_INVALID}: {field_name} must be relative to the workspace root")
    if any(part == ".." for part in path.parts):
        raise ValueError(f"{YIU_PATH_INVALID}: {field_name} must not traverse outside the workspace root")
    return path


__all__ = [
    "BASES",
    "SECONDARY_OBJECTIVE_LADDER",
    "normalize_optional_text",
    "normalize_yiu_sequence",
    "require_non_empty_text",
    "validate_workspace_relative_path",
]
