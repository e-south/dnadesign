"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/cli/common.py

Shared helpers for OPS CLI command modules.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

import click

from dnadesign.ops.catalog import RunbookCatalog, suggest_procedure_registry_ids

_PENDING_STDERR_MESSAGES: list[str] = []


class OpsCliContractError(click.ClickException):
    exit_code = 2


def raise_contract_error(message: str) -> None:
    raise OpsCliContractError(message)


def emit_stderr(message: str) -> None:
    text = str(message or "")
    if not text:
        return
    if not text.endswith("\n"):
        text += "\n"
    _PENDING_STDERR_MESSAGES.append(text)
    if sys.stderr is not getattr(sys, "__stderr__", None):
        sys.stderr.write(text)
        sys.stderr.flush()


def pop_pending_stderr_messages() -> tuple[str, ...]:
    messages = tuple(_PENDING_STDERR_MESSAGES)
    _PENDING_STDERR_MESSAGES.clear()
    return messages


def reset_pending_stderr_messages() -> None:
    _PENDING_STDERR_MESSAGES.clear()


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
    "OpsCliContractError",
    "append_registry_suggestions",
    "emit_stderr",
    "normalize_optional_filter",
    "pop_pending_stderr_messages",
    "raise_contract_error",
    "render_command",
    "reset_pending_stderr_messages",
]
