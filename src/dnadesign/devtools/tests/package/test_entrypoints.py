"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/package/test_entrypoints.py

Checks for installed command dependency diagnostics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import pytest

from dnadesign.devtools.package import entrypoints


def test_missing_optional_dependency_reports_install_profile(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError("No module named pandas", name="pandas")

    monkeypatch.setattr(entrypoints, "import_module", missing)
    with pytest.raises(SystemExit, match=r"dnadesign-tools\[usr\].*pandas"):
        entrypoints.usr()


def test_internal_import_failure_is_not_misreported_as_an_optional_dependency(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError("Internal module missing", name="dnadesign.usr.missing")

    monkeypatch.setattr(entrypoints, "import_module", missing)
    with pytest.raises(ModuleNotFoundError):
        entrypoints.usr()


def test_launcher_preserves_command_return_value(monkeypatch):
    class App:
        @staticmethod
        def main():
            return 7

    monkeypatch.setattr(entrypoints, "import_module", lambda name: App)
    assert entrypoints.usr() == 7
