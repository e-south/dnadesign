"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/registry.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Callable, Dict

import typer


@dataclass(frozen=True)
class CommandSpec:
    name: str
    help: str
    callback: Callable[..., None]


_CLI_REGISTRY: Dict[str, CommandSpec] = {}


def cli_command(name: str, help: str):
    def _wrap(func: Callable[..., None]) -> Callable[..., None]:
        if name in _CLI_REGISTRY:
            raise RuntimeError(f"CLI command '{name}' already registered")
        _CLI_REGISTRY[name] = CommandSpec(name=name, help=help, callback=func)
        return func

    return _wrap


def install_registered_commands(app: typer.Typer) -> None:
    for spec in _CLI_REGISTRY.values():
        app.command(name=spec.name, help=spec.help)(spec.callback)


def discover_commands(package: str = "dnadesign.opal.src.cli.commands") -> None:
    """
    Import all modules under the commands package so that @cli_command
    decorators run. Import errors are captured and deferred: we register a
    placeholder command that, when invoked, prints the original import error.
    """
    module = importlib.import_module(package)
    pkg_path = module.__path__  # type: ignore[attr-defined]

    failures: Dict[str, Exception] = {}

    for m in pkgutil.iter_modules(pkg_path):
        mod_name = f"{package}.{m.name}"
        try:
            importlib.import_module(mod_name)
        except Exception as e:
            failures[m.name] = e

    # Register placeholders for failures so other commands still work.
    for name, err in failures.items():
        placeholder_name = name.replace("_", "-")

        def _make_placeholder(n=name, exc=err):
            def _fail_cmd():
                raise typer.Exit(
                    code=1
                ) from None  # keeps Typer's clean message; full TB with OPAL_DEBUG=1

            return _fail_cmd

        # Only add if not already defined by a successful module
        if placeholder_name not in _CLI_REGISTRY:
            _CLI_REGISTRY[placeholder_name] = CommandSpec(
                name=placeholder_name,
                help="(unavailable due to import error; set OPAL_DEBUG=1 for details)",
                callback=_make_placeholder(),
            )
