"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/seqledger/tests/cli/commands/lifecycle/test_cli_materialize_module_layout.py

Layout contract tests for SeqLedger materialize command decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

import dnadesign.seqledger.src.cli.support.wiring.bindings as cli_bindings


def test_usr_cli_materialize_module_importable() -> None:
    module = importlib.import_module("dnadesign.seqledger.src.cli.commands.lifecycle.materialize")
    assert hasattr(module, "MaterializeDeps")
    assert hasattr(module, "cmd_materialize")


def test_usr_cli_materialize_command_delegates_to_materialize_module() -> None:
    source = inspect.getsource(cli_bindings)
    assert "lifecycle_commands.cmd_materialize(" in source
