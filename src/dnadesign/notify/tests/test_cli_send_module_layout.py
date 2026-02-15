"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_cli_send_module_layout.py

Layout contract tests for notify send command decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

import dnadesign.notify.cli as notify_cli


def test_notify_send_command_module_importable() -> None:
    assert importlib.import_module("dnadesign.notify.cli_commands.send")


def test_notify_cli_registers_send_command_from_module() -> None:
    source = inspect.getsource(notify_cli)
    assert "register_send_command(" in source
    assert '@app.command("send")' not in source
