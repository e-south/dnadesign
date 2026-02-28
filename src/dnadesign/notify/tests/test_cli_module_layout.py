"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_cli_module_layout.py

Module layout contract tests for Notify CLI command decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dnadesign.notify.cli import app


def _notify_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_notify_cli_command_modules_importable() -> None:
    assert importlib.import_module("dnadesign.notify.cli.commands.delivery")
    assert importlib.import_module("dnadesign.notify.cli.commands.delivery.send_cmd")
    assert importlib.import_module("dnadesign.notify.cli.commands.delivery.providers")
    assert importlib.import_module("dnadesign.notify.cli.commands.profile")
    assert importlib.import_module("dnadesign.notify.cli.commands.profile.init_cmd")
    assert importlib.import_module("dnadesign.notify.cli.commands.profile.wizard_cmd")
    assert importlib.import_module("dnadesign.notify.cli.commands.profile.show_cmd")
    assert importlib.import_module("dnadesign.notify.cli.commands.profile.doctor_cmd")
    assert importlib.import_module("dnadesign.notify.cli.commands.runtime")
    assert importlib.import_module("dnadesign.notify.cli.commands.runtime.watch_cmd")
    assert importlib.import_module("dnadesign.notify.cli.commands.runtime.spool_cmd")
    assert importlib.import_module("dnadesign.notify.cli.commands.setup")
    assert importlib.import_module("dnadesign.notify.cli.commands.setup.slack_cmd")
    assert importlib.import_module("dnadesign.notify.cli.commands.setup.webhook_cmd")
    assert importlib.import_module("dnadesign.notify.cli.commands.setup.resolve_events_cmd")
    assert importlib.import_module("dnadesign.notify.cli.commands.setup.list_workspaces_cmd")
    bindings = importlib.import_module("dnadesign.notify.cli.bindings")
    assert getattr(bindings, "__path__", None) is not None
    assert importlib.import_module("dnadesign.notify.cli.bindings.send")
    assert importlib.import_module("dnadesign.notify.cli.bindings.profile")
    assert importlib.import_module("dnadesign.notify.cli.bindings.setup")
    assert importlib.import_module("dnadesign.notify.cli.bindings.runtime")
    assert importlib.import_module("dnadesign.notify.cli.bindings.helpers")
    assert importlib.import_module("dnadesign.notify.cli.bindings.deps")
    assert importlib.import_module("dnadesign.notify.cli.bindings.deps.profile")
    assert importlib.import_module("dnadesign.notify.cli.bindings.deps.setup")
    assert importlib.import_module("dnadesign.notify.cli.bindings.deps.runtime")
    assert importlib.import_module("dnadesign.notify.cli.bindings.deps.send")
    assert importlib.import_module("dnadesign.notify.cli.bindings.registry")
    assert importlib.import_module("dnadesign.notify.cli.handlers.profile")
    assert importlib.import_module("dnadesign.notify.cli.handlers.profile.init_cmd")
    assert importlib.import_module("dnadesign.notify.cli.handlers.profile.wizard_cmd")
    assert importlib.import_module("dnadesign.notify.cli.handlers.profile.show_cmd")
    assert importlib.import_module("dnadesign.notify.cli.handlers.profile.doctor_cmd")
    assert importlib.import_module("dnadesign.notify.cli.handlers.setup")
    assert importlib.import_module("dnadesign.notify.cli.handlers.setup.slack_cmd")
    assert importlib.import_module("dnadesign.notify.cli.handlers.setup.webhook_cmd")
    assert importlib.import_module("dnadesign.notify.cli.handlers.setup.resolve_events_cmd")
    assert importlib.import_module("dnadesign.notify.cli.handlers.setup.list_workspaces_cmd")


def test_notify_legacy_top_level_cli_module_paths_are_not_importable() -> None:
    legacy_modules = [
        "dnadesign.notify.cli_bindings",
        "dnadesign.notify.cli_commands",
        "dnadesign.notify.cli_handlers",
        "dnadesign.notify.cli_resolve",
        "dnadesign.notify.cli.commands.send",
        "dnadesign.notify.cli.commands.usr_events",
        "dnadesign.notify.cli.commands.spool",
        "dnadesign.notify.cli.commands.providers",
    ]
    for module_path in legacy_modules:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_path)


def test_notify_cli_command_monolith_files_removed() -> None:
    commands_root = _notify_root() / "cli" / "commands"
    bindings_root = _notify_root() / "cli" / "bindings"
    assert not (commands_root / "profile.py").exists()
    assert not (commands_root / "setup.py").exists()
    assert not (commands_root / "send.py").exists()
    assert not (commands_root / "providers.py").exists()
    assert not (commands_root / "usr_events.py").exists()
    assert not (commands_root / "spool.py").exists()
    assert not (bindings_root / "deps.py").exists()


def test_notify_help_lists_usr_events_and_spool_groups() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "usr-events" in result.stdout
    assert "spool" in result.stdout
    assert "profile" in result.stdout


def test_notify_cli_registers_commands_from_bindings_module() -> None:
    source = inspect.getsource(importlib.import_module("dnadesign.notify.cli"))
    assert "register_notify_cli_bindings(" in source
    assert "_profile_init_impl(" not in source
    assert "_usr_events_watch_impl(" not in source


def test_notify_cli_router_does_not_define_command_options() -> None:
    source = inspect.getsource(importlib.import_module("dnadesign.notify.cli"))
    assert "typer.Option(" not in source
