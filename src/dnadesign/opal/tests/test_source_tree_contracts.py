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


def test_notebook_set_template_is_semantic_package_modules() -> None:
    template_file = OPAL_SOURCE_ROOT / "analysis" / "notebook_set_template.py"
    template_package = OPAL_SOURCE_ROOT / "analysis" / "notebook_set_template"

    assert not template_file.exists()
    assert template_package.is_dir()
    module_lengths = {
        path.name: len(path.read_text().splitlines())
        for path in template_package.glob("*.py")
        if path.name != "__init__.py"
    }
    assert {
        "_support.py",
        "campaign_cells.py",
        "cells.py",
        "details_cells.py",
        "renderer.py",
        "setup_cells.py",
        "visual_cells.py",
    }.issubset(module_lengths)
    assert max(module_lengths.values()) <= 180


def test_generated_notebook_templates_do_not_import_dashboard_internals() -> None:
    dashboard_package = OPAL_SOURCE_ROOT / "analysis" / "dashboard"
    template_packages = [
        OPAL_SOURCE_ROOT / "analysis" / "notebook_template",
        OPAL_SOURCE_ROOT / "analysis" / "notebook_set_template",
    ]

    assert dashboard_package.is_dir()
    for package in template_packages:
        for path in package.glob("*.py"):
            text = path.read_text()
            assert "analysis.dashboard" not in text
            assert "dnadesign.opal.src.analysis.dashboard" not in text
            assert "dnadesign.opal.notebooks.api.generated" in text or "notebooks.api" not in text
            assert "from dnadesign.opal.notebooks.api import" not in text


def test_notebook_api_has_separate_generated_and_progress_surfaces() -> None:
    api_package = OPAL_ROOT / "notebooks" / "api"
    generated_api = api_package / "generated.py"
    progress_api = api_package / "progress.py"
    aggregate_api = api_package / "__init__.py"

    assert generated_api.is_file()
    assert progress_api.is_file()
    assert aggregate_api.is_file()
    assert "analysis.dashboard" not in generated_api.read_text()
    assert "analysis.dashboard.api" in progress_api.read_text()
    assert "Compatibility aggregate" in aggregate_api.read_text()
    module_lengths = {path.name: len(path.read_text().splitlines()) for path in api_package.glob("*.py")}
    assert max(module_lengths.values()) <= 160


def test_analysis_root_has_no_facade_or_flat_helper_modules() -> None:
    analysis_package = OPAL_SOURCE_ROOT / "analysis"
    root_modules = sorted(path.name for path in analysis_package.glob("*.py") if path.name != "__init__.py")

    assert not (analysis_package / "facade.py").exists()
    assert root_modules == []
    init_text = (analysis_package / "__init__.py").read_text()
    assert "CampaignAnalysis" not in init_text
    assert "read_predictions" not in init_text
    assert "load_predictions_with_setpoint" not in init_text


def test_analysis_core_concepts_are_semantic_package_modules() -> None:
    analysis_package = OPAL_SOURCE_ROOT / "analysis"
    packages = {
        "campaign": {"analysis.py", "data.py", "loading.py"},
        "ledger": {"io.py", "predictions.py", "rounds.py", "setpoints.py"},
        "notebook_scope": {"resolution.py"},
    }

    for package_name, expected_modules in packages.items():
        package = analysis_package / package_name
        assert package.is_dir()
        assert (package / "__init__.py").is_file()
        module_lengths = {
            path.name: len(path.read_text().splitlines()) for path in package.glob("*.py") if path.name != "__init__.py"
        }
        assert expected_modules.issubset(module_lengths)
        assert max(module_lengths.values()) <= 220


def test_campaign_progress_analysis_is_semantic_package_modules() -> None:
    progress_file = OPAL_SOURCE_ROOT / "analysis" / "campaign_progress.py"
    progress_package = OPAL_SOURCE_ROOT / "analysis" / "campaign_progress"

    assert not progress_file.exists()
    assert progress_package.is_dir()
    module_lengths = {
        path.name: len(path.read_text().splitlines())
        for path in progress_package.glob("*.py")
        if path.name != "__init__.py"
    }
    assert {"content.py", "ledger.py", "models.py", "records.py"}.issubset(module_lengths)
    assert max(module_lengths.values()) <= 180


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


def test_round_stages_are_semantic_package_modules() -> None:
    stages_file = OPAL_SOURCE_ROOT / "runtime" / "round" / "stages.py"
    stages_package = OPAL_SOURCE_ROOT / "runtime" / "round" / "stages"

    assert not stages_file.exists()
    assert stages_package.is_dir()
    module_lengths = {
        path.name: len(path.read_text().splitlines())
        for path in stages_package.glob("*.py")
        if path.name != "__init__.py"
    }
    assert module_lengths
    assert max(module_lengths.values()) <= 260


def test_round_stages_package_imports_real_stage_entrypoints() -> None:
    module = importlib.import_module("dnadesign.opal.src.runtime.round.stages")

    assert module.stage_training.__name__ == "stage_training"
    assert module.stage_x_matrices.__name__ == "stage_x_matrices"
    assert module.stage_scoring.__name__ == "stage_scoring"
