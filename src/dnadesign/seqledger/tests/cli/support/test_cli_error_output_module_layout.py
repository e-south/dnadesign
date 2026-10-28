"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/seqledger/tests/cli/support/test_cli_error_output_module_layout.py

Layout contract tests for SeqLedger CLI error rendering decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

import dnadesign.seqledger.src.cli as usr_cli


def test_usr_cli_error_output_module_importable() -> None:
    module = importlib.import_module("dnadesign.seqledger.src.cli.commands.error_output")
    assert hasattr(module, "print_user_error")


def test_usr_cli_error_output_delegates_to_module() -> None:
    source = inspect.getsource(usr_cli)
    assert "error_output_commands.print_user_error(" in source
