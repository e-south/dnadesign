"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/package/test_entrypoints.py

Checks for installed command dependency diagnostics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import tomllib
from importlib.metadata import requires
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from dnadesign import _entrypoints as entrypoints
from dnadesign.devtools.tests.support.dependencies import profile_requirements


def test_missing_optional_dependency_reports_install_profile(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError("No module named pandas", name="pandas")

    monkeypatch.setattr(entrypoints, "import_module", missing)
    with pytest.raises(SystemExit, match=r"dnadesign-tools\[usr\].*pandas"):
        entrypoints.usr()


@pytest.mark.parametrize(
    "command", ["ops", "notify", "construct", "folding", "cluster", "latentdna", "baserender", "cruncher", "junction"]
)
def test_cpu_command_recommends_its_own_runtime_profile(monkeypatch, command):
    def missing(name):
        raise ModuleNotFoundError("No module named typer", name="typer")

    monkeypatch.setattr(entrypoints, "import_module", missing)
    with pytest.raises(SystemExit, match=rf"dnadesign-tools\[{command}\]"):
        getattr(entrypoints, command)()


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


@pytest.mark.parametrize("during_execution", [False, True])
def test_unknown_dependency_keeps_original_error_and_context(monkeypatch, during_execution):
    error = ModuleNotFoundError("Plugin dependency missing", name="task_specific_dependency")

    def missing(*args):
        raise error

    class App:
        main = staticmethod(missing)

    monkeypatch.setattr(entrypoints, "import_module", (lambda name: App) if during_execution else missing)
    with pytest.raises(ModuleNotFoundError) as caught:
        entrypoints.usr()
    assert caught.value is error
    assert caught.value.__traceback__ is not None


def test_known_lazy_optional_dependency_keeps_actionable_advice(monkeypatch):
    class App:
        @staticmethod
        def main():
            raise ModuleNotFoundError("No module named pandas", name="pandas")

    monkeypatch.setattr(entrypoints, "import_module", lambda name: App)
    with pytest.raises(SystemExit, match=r"dnadesign-tools\[usr\].*pandas"):
        entrypoints.usr()


@pytest.mark.parametrize("dependency", ["matplotlib", "torch", "flash_attn"])
def test_known_dependency_not_in_selected_profile_is_not_misattributed(monkeypatch, dependency):
    def missing(name):
        raise ModuleNotFoundError("Unexpected optional import", name=dependency)

    monkeypatch.setattr(entrypoints, "import_module", missing)
    with pytest.raises(ModuleNotFoundError):
        entrypoints.usr()


def test_missing_submodule_is_not_misreported_as_missing_distribution(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError("Incompatible pandas API", name="pandas.missing_api")

    monkeypatch.setattr(entrypoints, "import_module", missing)
    with pytest.raises(ModuleNotFoundError):
        entrypoints.usr()


def test_diagnostic_profiles_actually_supply_the_named_dependencies():
    root = next(path for path in Path(__file__).resolve().parents if (path / "pyproject.toml").is_file())
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    aliases = {
        "Bio": "biopython",
        "yaml": "pyyaml",
        "sklearn": "scikit-learn",
        "markdown_it": "markdown-it-py",
        "PIL": "pillow",
        "Levenshtein": "python-levenshtein",
        "umap": "umap-learn",
        "vl_convert": "vl-convert-python",
        "RNA": "viennarna",
    }

    for extra, modules in entrypoints._PROFILE_IMPORTS.items():
        names = {canonicalize_name(aliases.get(module, module)) for module in modules}
        supplied = {canonicalize_name(Requirement(value).name) for value in profile_requirements(project, extra)}
        if "pydantic" in supplied:
            assert "typing-extensions" in {canonicalize_name(Requirement(value).name) for value in requires("pydantic")}
            supplied.add("typing-extensions")
        assert names <= supplied


@pytest.mark.parametrize("command", ["latentdna", "cruncher", "opal"])
def test_missing_notebook_runtime_reports_notebook_profile(monkeypatch, command):
    def missing(name):
        raise ModuleNotFoundError("No module named marimo", name="marimo")

    monkeypatch.setattr(entrypoints, "import_module", missing)
    with pytest.raises(SystemExit, match=rf"dnadesign-tools\[{command}-notebook\]"):
        getattr(entrypoints, command)()
