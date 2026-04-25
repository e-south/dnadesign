"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/cli/commands/datasets/test_cli_datasets_package_module.py

Layout contract tests for CLI dataset helper package decomposition.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

import dnadesign.usr.src.cli as usr_cli
import dnadesign.usr.src.cli.support.resolution.dataset_targets as dataset_target_support
from dnadesign.usr.src.cli.commands.query import read as query_read_commands


def test_usr_cli_datasets_package_importable() -> None:
    assert importlib.import_module("dnadesign.usr.src.cli.commands.datasets")
    assert importlib.import_module("dnadesign.usr.src.cli.commands.datasets.catalog")
    assert importlib.import_module("dnadesign.usr.src.cli.commands.datasets.resolution")


def test_usr_cli_and_query_commands_use_dataset_helper_package() -> None:
    cli_source = inspect.getsource(usr_cli)
    helper_source = inspect.getsource(dataset_target_support)
    query_read_source = inspect.getsource(query_read_commands)

    assert "dataset_target_support" in cli_source
    assert "from ...commands import datasets as dataset_commands" in helper_source
    assert "from ..datasets import list_datasets, resolve_dataset_name_interactive" in query_read_source
