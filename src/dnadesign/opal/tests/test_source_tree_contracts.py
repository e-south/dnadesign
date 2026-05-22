from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

OPAL_ROOT = Path("src/dnadesign/opal")
OPAL_SOURCE_ROOT = OPAL_ROOT / "src"


def test_opal_package_root_has_no_ad_hoc_python_modules() -> None:
    root_modules = sorted(path.name for path in OPAL_ROOT.glob("*.py") if path.name != "__init__.py")

    assert root_modules == []
    assert (OPAL_SOURCE_ROOT / "cli").is_dir()
    assert (OPAL_SOURCE_ROOT / "analysis" / "dashboard").is_dir()


def test_opal_code_lives_under_source_or_declared_nonruntime_surfaces() -> None:
    allowed_roots = {
        OPAL_ROOT / "__init__.py",
        OPAL_ROOT / "api",
        OPAL_ROOT / "notebooks",
        OPAL_ROOT / "tests",
        OPAL_SOURCE_ROOT,
    }
    leaked = []
    for path in OPAL_ROOT.rglob("*.py"):
        if any(path == root or root in path.parents for root in allowed_roots):
            continue
        leaked.append(path.as_posix())

    assert leaked == []


def test_opal_console_script_targets_public_package_entrypoint() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    assert pyproject["project"]["scripts"]["opal"] == "dnadesign.opal:main"


def test_sfxi_public_api_namespace_is_declared_public_surface() -> None:
    assert (OPAL_ROOT / "api" / "__init__.py").is_file()
    assert (OPAL_ROOT / "api" / "sfxi.py").is_file()
    assert not (OPAL_SOURCE_ROOT / "api").exists()


def test_notebook_components_are_semantic_package_modules() -> None:
    component_file = OPAL_SOURCE_ROOT / "analysis" / "notebook_components.py"
    component_package = OPAL_SOURCE_ROOT / "analysis" / "notebook_components"

    assert not component_file.exists()
    assert component_package.is_dir()
    module_lengths = {
        path.name: len(path.read_text().splitlines())
        for path in component_package.glob("*.py")
        if path.name != "__init__.py"
    }
    assert module_lengths
    assert max(module_lengths.values()) <= 360


def test_notebook_template_is_semantic_package_modules() -> None:
    template_file = OPAL_SOURCE_ROOT / "analysis" / "notebook_template.py"
    template_package = OPAL_SOURCE_ROOT / "analysis" / "notebook_template"

    assert not template_file.exists()
    assert template_package.is_dir()
    module_lengths = {
        path.name: len(path.read_text().splitlines())
        for path in template_package.glob("*.py")
        if path.name != "__init__.py"
    }
    assert module_lengths
    assert max(module_lengths.values()) <= 220


def test_label_history_is_semantic_package_modules() -> None:
    history_file = OPAL_SOURCE_ROOT / "storage" / "label_history.py"
    history_package = OPAL_SOURCE_ROOT / "storage" / "label_history"

    assert not history_file.exists()
    assert history_package.is_dir()
    module_lengths = {
        path.name: len(path.read_text().splitlines())
        for path in history_package.glob("*.py")
        if path.name != "__init__.py"
    }
    assert module_lengths
    assert max(module_lengths.values()) <= 260


def test_ingest_y_command_is_semantic_package_modules() -> None:
    command_file = OPAL_SOURCE_ROOT / "cli" / "commands" / "ingest_y.py"
    command_package = OPAL_SOURCE_ROOT / "cli" / "commands" / "ingest_y"

    assert not command_file.exists()
    assert command_package.is_dir()
    module_lengths = {
        path.name: len(path.read_text().splitlines())
        for path in command_package.glob("*.py")
        if path.name != "__init__.py"
    }
    assert module_lengths
    assert max(module_lengths.values()) <= 280


def test_ingest_y_command_package_imports_real_command() -> None:
    module = importlib.import_module("dnadesign.opal.src.cli.commands.ingest_y")

    assert module.cmd_ingest_y.__name__ == "cmd_ingest_y"
