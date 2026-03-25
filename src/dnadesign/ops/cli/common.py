"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/cli/common.py

Shared helpers for OPS CLI command modules.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Sequence

from dnadesign.ops.catalog import RunbookCatalog, suggest_procedure_registry_ids


def append_registry_suggestions(*, message: str, catalog: RunbookCatalog, registry_id: str) -> str:
    suggestions = suggest_procedure_registry_ids(catalog, registry_id)
    if suggestions:
        message += "\nDid you mean:\n" + "\n".join(f"- {candidate}" for candidate in suggestions)
    return message


def normalize_optional_filter(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def render_command(parts: Sequence[str]) -> str:
    return " ".join(parts)


__all__ = [
    "append_registry_suggestions",
    "normalize_optional_filter",
    "render_command",
]
