"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_cli_sync_args_builder_layout.py

Layout contract tests for sync argument builder wiring in USR CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import inspect

import dnadesign.usr.src.cli as usr_cli


def test_usr_cli_sync_registration_uses_shared_ctx_args_builder() -> None:
    source = inspect.getsource(usr_cli)
    assert "sync_args_builder=_ctx_args" in source


def test_usr_cli_no_local_sync_args_shim() -> None:
    source = inspect.getsource(usr_cli)
    assert "def _sync_args(" not in source
