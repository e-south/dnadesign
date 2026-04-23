"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_cli_datasets_package_module.py

Layout contract tests for CLI dataset helper package decomposition.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

import dnadesign.usr.src.cli as usr_cli
from dnadesign.usr.src.cli_commands.query import read as query_read_commands


def test_usr_cli_datasets_package_importable() -> None:
    assert importlib.import_module("dnadesign.usr.src.cli_commands.datasets")
    assert importlib.import_module("dnadesign.usr.src.cli_commands.datasets.catalog")
    assert importlib.import_module("dnadesign.usr.src.cli_commands.datasets.resolution")


def test_usr_cli_and_query_commands_use_dataset_helper_package() -> None:
    cli_source = inspect.getsource(usr_cli)
    query_read_source = inspect.getsource(query_read_commands)

    assert "from .cli_commands import datasets as dataset_commands" in cli_source
    assert "from ..datasets import list_datasets, resolve_dataset_name_interactive" in query_read_source
