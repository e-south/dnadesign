"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_cli_event_output_module.py

Tests for CLI event-output format registry helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.usr.src.cli_event_output import emit_event_line, register_event_output_format
from dnadesign.usr.src.errors import SequencesError


def test_emit_event_line_supports_raw_and_json() -> None:
    payload = '{"x":1}'
    assert emit_event_line(payload, "raw") == payload
    assert emit_event_line(payload, "json") == payload


def test_emit_event_line_rejects_unknown_format() -> None:
    with pytest.raises(SequencesError, match="format must be one of"):
        emit_event_line('{"x":1}', "yaml")


def test_register_event_output_format_rejects_duplicate_name() -> None:
    register_event_output_format("unit_custom_event", lambda line: line)
    with pytest.raises(SequencesError, match="format 'unit_custom_event' is already registered"):
        register_event_output_format("unit_custom_event", lambda line: line)

