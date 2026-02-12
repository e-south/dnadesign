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

import dnadesign.notify.cli as notify_cli


def test_notify_watch_runtime_module_importable() -> None:
    assert importlib.import_module("dnadesign.notify.cli_runtime")


def test_notify_cli_watch_and_spool_delegate_to_runtime_module() -> None:
    source = inspect.getsource(notify_cli)
    assert "run_usr_events_watch(" in source
    assert "run_spool_drain(" in source
