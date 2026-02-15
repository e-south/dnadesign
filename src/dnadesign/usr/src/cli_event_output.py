"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/src/cli_event_output.py

Registry-backed event output formatting for USR CLI tail streams.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections.abc import Callable

from .errors import SequencesError

EventOutputEmitter = Callable[[str], str]

_EVENT_OUTPUT_FORMATS: dict[str, EventOutputEmitter] = {}


def _normalize_format_name(name: str | None) -> str:
    text = str(name or "").strip().lower()
    if not text:
        raise SequencesError("format name must be a non-empty string")
    return text


def supported_event_output_formats() -> tuple[str, ...]:
    return tuple(sorted(_EVENT_OUTPUT_FORMATS))


def register_event_output_format(name: str, emitter: EventOutputEmitter) -> None:
    format_name = _normalize_format_name(name)
    if format_name in _EVENT_OUTPUT_FORMATS:
        raise SequencesError(f"format '{format_name}' is already registered")
    if not callable(emitter):
        raise SequencesError("event output emitter must be callable")
    _EVENT_OUTPUT_FORMATS[format_name] = emitter


def emit_event_line(line: str, fmt: str) -> str | None:
    text = str(line or "").strip()
    if not text:
        return None
    format_name = _normalize_format_name(fmt)
    emitter = _EVENT_OUTPUT_FORMATS.get(format_name)
    if emitter is None:
        allowed = ", ".join(supported_event_output_formats())
        raise SequencesError(f"format must be one of: {allowed}.")
    return emitter(text)


def _emit_raw(text: str) -> str:
    return text


def _emit_json(text: str) -> str:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SequencesError(f"Invalid JSONL event line: {exc}") from exc
    return json.dumps(payload, separators=(",", ":"))


register_event_output_format("raw", _emit_raw)
register_event_output_format("json", _emit_json)
