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
    extras = tomllib.loads((root / "pyproject.toml").read_text())["project"]["optional-dependencies"]
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

    def distributions(extra):
        names = set()
        for value in extras[extra]:
            requirement = Requirement(value)
            if requirement.name == "dnadesign-tools":
                for parent in requirement.extras:
                    names.update(distributions(parent))
            else:
                names.add(canonicalize_name(requirement.name))
        return names

    for extra, modules in entrypoints._PROFILE_IMPORTS.items():
        names = {canonicalize_name(aliases.get(module, module)) for module in modules}
        supplied = distributions(extra)
        if "pydantic" in supplied:
            assert "typing-extensions" in {canonicalize_name(Requirement(value).name) for value in requires("pydantic")}
            supplied.add("typing-extensions")
        assert names <= supplied
