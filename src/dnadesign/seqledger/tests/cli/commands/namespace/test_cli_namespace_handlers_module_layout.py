"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/seqledger/tests/cli/commands/namespace/test_cli_namespace_handlers_module_layout.py

Layout contract tests for SeqLedger namespace handler decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

import dnadesign.seqledger.src.cli.support.wiring.bindings as cli_bindings


def test_usr_cli_namespace_handlers_module_importable() -> None:
    module = importlib.import_module("dnadesign.seqledger.src.cli.commands.namespace")
    assert hasattr(module, "NamespaceDeps")
    assert hasattr(module, "cmd_namespace_list")
    assert hasattr(module, "cmd_namespace_show")
    assert hasattr(module, "cmd_namespace_register")


def test_usr_cli_namespace_handlers_delegate_to_module() -> None:
    source = inspect.getsource(cli_bindings)
    assert "namespace_commands.cmd_namespace_list(" in source
    assert "namespace_commands.cmd_namespace_show(" in source
    assert "namespace_commands.cmd_namespace_register(" in source
