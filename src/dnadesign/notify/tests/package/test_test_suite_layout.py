"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/package/test_test_suite_layout.py

Contract tests for notify test-suite decomposition and file layout.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def _tests_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_notify_cli_profiles_monolith_removed() -> None:
    tests_root = _tests_root()
    assert not (tests_root / "test_cli_profiles.py").exists()


def test_notify_cli_profile_tests_split_by_command_surface() -> None:
    tests_root = _tests_root()
    assert (tests_root / "cli" / "test_profile_init.py").exists()
    assert (tests_root / "cli" / "test_profile_wizard.py").exists()
    assert (tests_root / "cli" / "test_setup.py").exists()
    assert (tests_root / "cli" / "test_profile_doctor.py").exists()
    assert (tests_root / "cli" / "test_profile_runtime_defaults.py").exists()


def test_notify_tests_are_grouped_by_runtime_domain() -> None:
    tests_root = _tests_root()
    expected_domains = {
        "cli",
        "delivery",
        "docs",
        "events",
        "hpc",
        "package",
        "profiles",
        "providers",
        "runtime",
        "tool_events",
    }
    assert {path.name for path in tests_root.iterdir() if path.is_dir()} >= expected_domains
