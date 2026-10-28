"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/seqledger/tests/cli/commands/read_views/test_cli_read_views_module_layout.py

Layout contract tests for SeqLedger read view command decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

import dnadesign.seqledger.src.cli.support.wiring.bindings as cli_bindings


def test_usr_cli_read_views_module_importable() -> None:
    module = importlib.import_module("dnadesign.seqledger.src.cli.commands.read_views")
    assert hasattr(module, "cmd_head")
    assert hasattr(module, "cmd_cols")
    assert hasattr(module, "cmd_describe")
    assert hasattr(module, "cmd_cell")


def test_usr_cli_read_view_commands_delegate_to_read_views_module() -> None:
    source = inspect.getsource(cli_bindings)
    assert "read_views_commands.cmd_head(" in source
    assert "read_views_commands.cmd_cols(" in source
    assert "read_views_commands.cmd_describe(" in source
    assert "read_views_commands.cmd_cell(" in source
