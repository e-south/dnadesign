"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_test_suite_layout.py

Contract tests for notify test-suite decomposition and file layout.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def _tests_root() -> Path:
    return Path(__file__).resolve().parent


def test_notify_cli_profiles_monolith_removed() -> None:
    tests_root = _tests_root()
    assert not (tests_root / "test_cli_profiles.py").exists()


def test_notify_cli_profile_tests_split_by_command_surface() -> None:
    tests_root = _tests_root()
    assert (tests_root / "test_cli_profile_init.py").exists()
    assert (tests_root / "test_cli_profile_wizard.py").exists()
    assert (tests_root / "test_cli_setup.py").exists()
    assert (tests_root / "test_cli_profile_doctor.py").exists()
    assert (tests_root / "test_cli_profile_runtime_defaults.py").exists()
