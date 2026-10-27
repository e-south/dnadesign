"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/cli/test_cli_main_lazy_registration.py

Coverage for DenseGen CLI lazy command scope helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.densegen.src.cli import main as dense_main


def test_command_scope_from_argv_skips_global_config_flag() -> None:
    scope = dense_main._command_scope_from_argv(["--config", "config.yaml", "notebook", "run"])
    assert scope == "notebook"


def test_command_scope_from_argv_handles_short_config_flag() -> None:
    scope = dense_main._command_scope_from_argv(["-c", "config.yaml", "plot"])
    assert scope == "plot"


def test_command_scope_from_argv_returns_none_for_help_only() -> None:
    scope = dense_main._command_scope_from_argv(["--help"])
    assert scope is None


def test_command_scope_from_argv_preserves_top_level_command_before_help() -> None:
    scope = dense_main._command_scope_from_argv(["run", "--help"])
    assert scope == "run"


def test_registration_targets_for_notebook_scope_is_minimal() -> None:
    targets = dense_main._registration_targets_for_scope("notebook")
    assert targets == {"notebook"}


def test_registration_targets_for_unknown_scope_falls_back_to_full_set() -> None:
    targets = dense_main._registration_targets_for_scope("unknown-command")
    assert targets == dense_main._ALL_REGISTRATION_TARGETS


def test_patch_typer_testing_get_command_requires_testing_hook() -> None:
    class _FakeTyperTesting:
        pass

    with pytest.raises(RuntimeError, match=r"typer\.testing\._get_command"):
        dense_main._patch_typer_testing_get_command(
            lambda _typer_instance: object(),
            typer_testing_module=_FakeTyperTesting(),
        )
