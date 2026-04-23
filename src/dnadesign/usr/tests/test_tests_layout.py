"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_tests_layout.py

Test layout contract for USR package decomposition.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from .tests_layout_inventory import (
    ROOT_TEST_MODULES,
    ROOT_TEST_SUPPORT_FILES,
    TOP_LEVEL_TEST_PACKAGES,
)


def test_usr_tests_root_contains_only_sanctioned_files_and_packages() -> None:
    tests_root = Path(__file__).resolve().parent
    actual_files = {path.name for path in tests_root.glob("*.py")}
    actual_packages = {path.name for path in tests_root.iterdir() if path.is_dir() and (path / "__init__.py").exists()}

    assert actual_files == ROOT_TEST_SUPPORT_FILES | ROOT_TEST_MODULES
    assert actual_packages == TOP_LEVEL_TEST_PACKAGES


def test_usr_test_family_packages_are_nonempty() -> None:
    tests_root = Path(__file__).resolve().parent

    for package_name in TOP_LEVEL_TEST_PACKAGES:
        package_root = tests_root / package_name
        actual_files = {path.name for path in package_root.glob("*.py")}

        assert "__init__.py" in actual_files
        assert any(name.startswith("test_") for name in actual_files if name != "__init__.py")
