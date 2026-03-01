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
    assert importlib.import_module("dnadesign.notify.cli.commands.delivery")
    assert importlib.import_module("dnadesign.notify.cli.commands.delivery.send_cmd")
    assert importlib.import_module("dnadesign.notify.cli.handlers.send")


def test_notify_cli_registers_send_command_from_module() -> None:
    router_source = inspect.getsource(notify_cli)
    bindings_source = inspect.getsource(importlib.import_module("dnadesign.notify.cli.bindings"))
    registry_source = inspect.getsource(importlib.import_module("dnadesign.notify.cli.bindings.registry"))
    send_bindings_source = inspect.getsource(importlib.import_module("dnadesign.notify.cli.bindings.send"))
    assert "register_notify_cli_bindings(" in router_source
    assert "_register_notify_cli_bindings(" in bindings_source
    assert "register_send_command(" in registry_source
    assert "run_send_command(" in send_bindings_source
    assert '@app.command("send")' not in router_source
