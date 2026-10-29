"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/cli/dispatch.py

Lazy root-command dispatch for OPS CLI subcommand groups.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib

import click
from typer.core import TyperGroup

_COMMAND_MODULES: dict[str, str] = {
    "catalog": "dnadesign.ops.cli.commands.catalog",
    "progress": "dnadesign.ops.cli.commands.progress",
    "runbook": "dnadesign.ops.cli.commands.runbook",
}


class LazyGroup(TyperGroup):
    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted(_COMMAND_MODULES)

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        module_path = _COMMAND_MODULES.get(cmd_name)
        if module_path is None:
            return None
        module = importlib.import_module(module_path)
        loader = getattr(module, "get_click_command", None)
        if loader is None or not callable(loader):
            raise RuntimeError(f"OPS CLI command module is missing get_click_command(): {module_path}")
        command = loader()
        command.name = cmd_name
        return command


__all__ = ["LazyGroup"]
