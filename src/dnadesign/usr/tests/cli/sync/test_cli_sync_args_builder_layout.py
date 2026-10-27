"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/cli/sync/test_cli_sync_args_builder_layout.py

Layout contract tests for sync argument builder wiring in USR CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import inspect

import dnadesign.usr.src.cli as usr_cli
import dnadesign.usr.src.cli.support.wiring.registration as cli_registration_support


def test_usr_cli_sync_registration_uses_shared_ctx_args_builder() -> None:
    assert usr_cli._ctx_args is cli_registration_support.ctx_args
    source = inspect.getsource(cli_registration_support.register_cli_surface)
    assert "sync_args_builder=ctx_args_builder" in source


def test_usr_cli_no_local_sync_args_shim() -> None:
    source = inspect.getsource(usr_cli)
    assert "def _sync_args(" not in source
