"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/cli/dynamic_inputs.py

Shared metadata-driven input parsing and rendering helpers for OPS status
commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .common import render_command


class StatusInputField(Protocol):
    name: str
    cli_flag: str
    placeholder: str
    summary: str


def render_progress_show_command(*, registry_id: str, required_inputs: Sequence[StatusInputField]) -> str:
    parts = ["uv", "run", "ops", "progress", "show", registry_id]
    for field in required_inputs:
        parts.extend((field.cli_flag, field.placeholder))
    return render_command(parts)


def required_input_lines(*, label: str, required_inputs: Sequence[StatusInputField]) -> tuple[str, ...]:
    if not required_inputs:
        return ()
    lines = [f"Required inputs for {label}:"]
    for field in required_inputs:
        lines.append(f"- {field.cli_flag} {field.placeholder}: {field.summary}")
    return tuple(lines)


def optional_input_lines(optional_inputs: Sequence[StatusInputField]) -> tuple[str, ...]:
    if not optional_inputs:
        return ()
    lines = ["Also accepted:"]
    for field in optional_inputs:
        lines.append(f"- {field.cli_flag}: {field.summary}")
    return tuple(lines)


def parse_status_input_tokens(
    *,
    extra_args: Sequence[str],
    input_items: Sequence[str],
    input_schema: Sequence[StatusInputField],
) -> dict[str, object]:
    inputs_by_flag = {field.cli_flag: field for field in input_schema}
    inputs_by_name = {field.name: field for field in input_schema}
    resolved_inputs: dict[str, object] = {}
    tokens = list(extra_args)
    index = 0
    while index < len(tokens):
        token = str(tokens[index]).strip()
        if not token:
            index += 1
            continue
        if not token.startswith("--"):
            raise ValueError(f"unexpected argument for progress show: {token}")
        if token == "--":
            index += 1
            continue
        if "=" in token:
            flag, value = token.split("=", maxsplit=1)
            index += 1
        else:
            flag = token
            if index + 1 >= len(tokens):
                raise ValueError(f"{flag} requires a value")
            value = str(tokens[index + 1])
            index += 2
        field = inputs_by_flag.get(flag)
        if field is None:
            raise ValueError(f"unknown progress input flag: {flag}")
        if field.name in resolved_inputs:
            raise ValueError(f"duplicate progress input: {field.cli_flag}")
        resolved_inputs[field.name] = value

    for item in input_items:
        if "=" not in item:
            raise ValueError("--input expects key=value")
        name, value = item.split("=", maxsplit=1)
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("--input expects a non-empty key")
        field = inputs_by_name.get(normalized_name)
        if field is None:
            raise ValueError(f"unknown progress input key: {normalized_name}")
        if field.name in resolved_inputs:
            raise ValueError(f"duplicate progress input: {field.cli_flag}")
        resolved_inputs[field.name] = value
    return resolved_inputs


__all__ = [
    "optional_input_lines",
    "parse_status_input_tokens",
    "render_progress_show_command",
    "required_input_lines",
]
