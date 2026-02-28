"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_cli_watch_runtime_layout.py

Layout contract tests for notify watch/spool runtime decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import dnadesign.notify.cli as notify_cli


def _notify_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_notify_watch_runtime_module_importable() -> None:
    assert importlib.import_module("dnadesign.notify.runtime.runner")
    assert importlib.import_module("dnadesign.notify.runtime.watch_runner")
    assert importlib.import_module("dnadesign.notify.runtime.watch_runner_contract")
    assert importlib.import_module("dnadesign.notify.runtime.watch_runner_resolution")
    assert importlib.import_module("dnadesign.notify.runtime.watch_events")
    assert importlib.import_module("dnadesign.notify.runtime.watch_delivery")
    assert importlib.import_module("dnadesign.notify.runtime.spool_runner")
    assert importlib.import_module("dnadesign.notify.runtime.watch")
    assert importlib.import_module("dnadesign.notify.runtime.cursor")
    assert importlib.import_module("dnadesign.notify.runtime.spool")
    assert importlib.import_module("dnadesign.notify.cli.handlers.runtime")
    assert importlib.import_module("dnadesign.notify.cli.handlers.runtime.watch_cmd")
    assert importlib.import_module("dnadesign.notify.cli.handlers.runtime.spool_cmd")


def test_notify_runtime_handler_monolith_removed() -> None:
    handlers_root = _notify_root() / "cli" / "handlers"
    assert not (handlers_root / "runtime.py").exists()


def test_notify_cli_watch_and_spool_delegate_to_runtime_module() -> None:
    router_source = inspect.getsource(notify_cli)
    bindings_source = inspect.getsource(importlib.import_module("dnadesign.notify.cli.bindings"))
    registry_source = inspect.getsource(importlib.import_module("dnadesign.notify.cli.bindings.registry"))
    runtime_bindings_source = inspect.getsource(importlib.import_module("dnadesign.notify.cli.bindings.runtime"))
    assert "register_notify_cli_bindings(" in router_source
    assert "_register_notify_cli_bindings(" in bindings_source
    assert "register_usr_events_watch_command(" in registry_source
    assert "register_spool_drain_command(" in registry_source
    assert "run_usr_events_watch_command(" in runtime_bindings_source
    assert "run_spool_drain_command(" in runtime_bindings_source
