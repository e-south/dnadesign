"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_cli_deps_module_layout.py

Layout contract tests for CLI dependency builder decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
import inspect

import dnadesign.usr.src.cli as usr_cli


def test_usr_cli_deps_module_importable() -> None:
    module = importlib.import_module("dnadesign.usr.src.cli_commands.deps")
    assert hasattr(module, "build_read_view_deps")
    assert hasattr(module, "build_runtime_deps")
    assert hasattr(module, "build_materialize_deps")
    assert hasattr(module, "build_maintenance_deps")
    assert hasattr(module, "build_merge_deps")
    assert hasattr(module, "build_namespace_deps")
    assert hasattr(module, "build_tooling_deps")


def test_usr_cli_dependency_builders_delegate_to_module() -> None:
    source = inspect.getsource(usr_cli)
    assert "deps_commands.build_read_view_deps(" in source
    assert "deps_commands.build_runtime_deps(" in source
    assert "deps_commands.build_materialize_deps(" in source
    assert "deps_commands.build_maintenance_deps(" in source
    assert "deps_commands.build_merge_deps(" in source
    assert "deps_commands.build_namespace_deps(" in source
    assert "deps_commands.build_tooling_deps(" in source
