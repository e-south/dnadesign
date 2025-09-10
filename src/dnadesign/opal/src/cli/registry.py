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
    module = importlib.import_module(package)
    pkg_path = module.__path__  # type: ignore[attr-defined]
    for m in pkgutil.iter_modules(pkg_path):
        importlib.import_module(f"{package}.{m.name}")
