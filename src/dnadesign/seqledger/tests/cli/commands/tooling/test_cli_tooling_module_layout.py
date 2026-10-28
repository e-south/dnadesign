"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/seqledger/tests/cli/commands/tooling/test_cli_tooling_module_layout.py

Layout contract tests for SeqLedger tooling command decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

import dnadesign.seqledger.src.cli.support.wiring.bindings as cli_bindings


def test_usr_cli_tooling_module_importable() -> None:
    module = importlib.import_module("dnadesign.seqledger.src.cli.commands.tooling")
    assert hasattr(module, "ToolingDeps")
    assert hasattr(module, "register_ops_commands")
    assert hasattr(module, "cmd_make_mock")
    assert hasattr(module, "cmd_add_demo")
    assert importlib.import_module("dnadesign.seqledger.src.cli.commands.tooling.cli")
    assert importlib.import_module("dnadesign.seqledger.src.cli.commands.tooling.dev")


def test_usr_cli_tooling_commands_delegate_to_module() -> None:
    source = inspect.getsource(cli_bindings)
    assert "tooling_commands.cmd_make_mock(" in source
    assert "tooling_commands.cmd_add_demo(" in source
