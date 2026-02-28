"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_cli_namespace_handlers_module_layout.py

Layout contract tests for USR namespace handler decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

import dnadesign.usr.src.cli as usr_cli


def test_usr_cli_namespace_handlers_module_importable() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.namespace_handlers")
    assert hasattr(module, "NamespaceDeps")
    assert hasattr(module, "cmd_namespace_list")
    assert hasattr(module, "cmd_namespace_show")
    assert hasattr(module, "cmd_namespace_register")


def test_usr_cli_namespace_handlers_delegate_to_module() -> None:
    source = inspect.getsource(usr_cli)
    assert "namespace_handlers_commands.cmd_namespace_list(" in source
    assert "namespace_handlers_commands.cmd_namespace_show(" in source
    assert "namespace_handlers_commands.cmd_namespace_register(" in source
