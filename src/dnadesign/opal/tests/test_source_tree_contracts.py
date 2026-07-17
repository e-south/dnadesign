"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/test_source_tree_contracts.py

Regression tests for source tree OPAL.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
import importlib
import tomllib
from pathlib import Path

import yaml

from dnadesign.opal.src.config.loader import load_config

OPAL_ROOT = Path("src/dnadesign/opal")
OPAL_SOURCE_ROOT = OPAL_ROOT / "src"


def _read_frontmatter(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"missing YAML frontmatter: {path}"
    payload = yaml.safe_load(text.split("---", 2)[1])
    assert isinstance(payload, dict), f"frontmatter must be a mapping: {path}"
    return payload


def test_checked_in_campaigns_are_explicit_modern_surfaces() -> None:
    campaign_configs = sorted((OPAL_ROOT / "campaigns").glob("*/configs/campaign.yaml"))
    expected = {
        "demo_gp_ei": ("opal_demo", "demo", "runnable"),
        "demo_gp_topn": ("opal_demo", "demo", "runnable"),
        "demo_rf_sfxi_topn": ("opal_demo", "demo", "runnable"),
        "secg_rmf_greedy": ("study_campaign", "study", "round0_complete"),
    }

    actual: dict[str, tuple[str, str, str]] = {}
    for path in campaign_configs:
        cfg = load_config(path)
        assert path.parents[1].name == cfg.campaign.slug
        readme = path.parents[1] / "README.md"
        frontmatter = _read_frontmatter(readme)
        assert frontmatter["surface"] == "opal_campaign"
        assert frontmatter["campaign_slug"] == cfg.campaign.slug
        assert sorted(candidate.name for candidate in path.parent.iterdir()) == ["campaign.yaml", "plots.yaml"]
        actual[cfg.campaign.slug] = (
            cfg.ownership.owner_scope,
            str(frontmatter["campaign_kind"]),
            str(frontmatter["runtime_status"]),
        )

    assert actual == expected


def test_campaign_index_declares_routing_frontmatter() -> None:
    frontmatter = _read_frontmatter(OPAL_ROOT / "campaigns" / "README.md")

    assert frontmatter["surface"] == "opal_campaign_index"
    assert frontmatter["status"] == "active"


def test_current_stress_campaign_config_has_no_sfxi_metric_or_brightness_aliases() -> None:
    config_path = OPAL_ROOT / "campaigns" / "secg_rmf_greedy" / "configs" / "campaign.yaml"
    text = config_path.read_text(encoding="utf-8").lower()

    assert "sfxi" not in text
    assert "vec8" not in text
    assert "brightness" not in text
    assert "stress_promoter_insert:v1" in text


def test_checked_in_campaigns_use_only_canonical_config_filenames() -> None:
    alternate_names = {"campaign.yml", "opal.yaml", "opal.yml"}
    alternates = [
        path.as_posix()
        for path in (OPAL_ROOT / "campaigns").rglob("*")
        if path.is_file() and path.name in alternate_names
    ]

    assert alternates == []


def _line_count(path: Path) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) >= 10 and lines[0] == '"""' and lines[1] == "-" * 80 and lines[8] == "-" * 80 and lines[9] == '"""':
        return len(lines) - 10
    return len(lines)


def _is_generated_campaign_notebook(path: Path) -> bool:
    try:
        relative = path.relative_to(OPAL_ROOT / "campaigns")
    except ValueError:
        return False
    if len(relative.parts) != 3 or relative.parts[1] != "notebooks":
        return False
    lines = path.read_text(encoding="utf-8").splitlines()
    has_marimo_preamble = bool(lines) and lines[0] == "import marimo"
    has_generated_marker = any(line.startswith("__generated_with") for line in lines[:5])
    return has_marimo_preamble and has_generated_marker


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
        if _is_generated_campaign_notebook(path):
            continue
        if any(path == root or root in path.parents for root in allowed_roots):
            continue
        leaked.append(path.as_posix())

    assert leaked == []


def test_opal_console_script_targets_public_package_entrypoint() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    assert pyproject["project"]["scripts"]["opal"] == "dnadesign.opal:main"


def test_opal_large_entrypoints_are_explicitly_budget_guarded() -> None:
    budgets = {
        OPAL_SOURCE_ROOT / "cli" / "commands" / "notebook.py": 660,
        OPAL_SOURCE_ROOT / "reporting" / "review.py": 680,
    }

    for path, max_lines in budgets.items():
        assert path.is_file()
        assert _line_count(path) <= max_lines


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
        path.name: _line_count(path) for path in component_package.glob("*.py") if path.name != "__init__.py"
    }
    assert module_lengths
    assert max(module_lengths.values()) <= 360


def test_notebook_template_is_semantic_package_modules() -> None:
    template_file = OPAL_SOURCE_ROOT / "analysis" / "notebook_template.py"
    template_package = OPAL_SOURCE_ROOT / "analysis" / "notebook_template"

    assert not template_file.exists()
    assert template_package.is_dir()
    module_lengths = {
        path.name: _line_count(path) for path in template_package.glob("*.py") if path.name != "__init__.py"
    }
    assert module_lengths
    assert max(module_lengths.values()) <= 220


def test_notebook_set_template_is_semantic_package_modules() -> None:
    template_file = OPAL_SOURCE_ROOT / "analysis" / "notebook_set_template.py"
    template_package = OPAL_SOURCE_ROOT / "analysis" / "notebook_set_template"

    assert not template_file.exists()
    assert template_package.is_dir()
    module_lengths = {
        path.name: _line_count(path) for path in template_package.glob("*.py") if path.name != "__init__.py"
    }
    assert {
        "_support.py",
        "baserender_cells.py",
        "baserender_record_cells.py",
        "baserender_scope_cells.py",
        "campaign_cells.py",
        "cells.py",
        "collection_cells.py",
        "details_cells.py",
        "renderer.py",
        "setup_cells.py",
        "visual_cells.py",
        "visual_panel_cells.py",
    }.issubset(module_lengths)
    assert max(module_lengths.values()) <= 190


def test_single_campaign_notebook_uses_public_campaign_review_template_seam() -> None:
    renderer_text = (OPAL_SOURCE_ROOT / "analysis" / "notebook_template" / "renderer.py").read_text()

    assert "from ..notebook_set_template import" in renderer_text
    assert "notebook_set_template.cells" not in renderer_text


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


def test_opal_source_does_not_import_reader_or_mutate_python_import_paths() -> None:
    violations: list[str] = []
    for path in sorted(OPAL_SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "reader" or alias.name.startswith("reader."):
                        violations.append(f"{path}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "reader" or module.startswith("reader."):
                    violations.append(f"{path}:{node.lineno}: from {module} import ...")
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Attribute)
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "sys"
                and node.func.value.attr == "path"
            ):
                violations.append(f"{path}:{node.lineno}: sys.path.{node.func.attr}(...)")

    assert violations == []


def test_notebook_api_has_separate_generated_and_progress_surfaces() -> None:
    import dnadesign.opal.notebooks.api as notebook_api
    from dnadesign.opal.notebooks.api import generated, progress

    api_package = OPAL_ROOT / "notebooks" / "api"
    generated_api = api_package / "generated.py"
    progress_api = api_package / "progress.py"
    aggregate_api = api_package / "__init__.py"

    assert generated_api.is_file()
    assert progress_api.is_file()
    assert aggregate_api.is_file()
    assert "analysis.dashboard" not in generated_api.read_text()
    assert "analysis.dashboard.api" in progress_api.read_text()
    assert "build_records_preview" not in aggregate_api.read_text()
    assert notebook_api.__all__ == ()
    module_lengths = {path.name: _line_count(path) for path in api_package.glob("*.py")}
    budgets = {"__init__.py": 10, "generated.py": 170, "progress.py": 80}
    assert set(module_lengths) == set(budgets)
    assert {name: length for name, length in module_lengths.items() if length > budgets[name]} == {}
    for name in [
        "build_notebook_campaign_set_metric_comparison_rows",
        "build_notebook_collection_visual_card_rows",
        "build_notebook_collection_visual_choices",
        "load_campaign_collection_manifest",
        "render_notebook_campaign_set_metric_comparison_image",
    ]:
        assert name in generated.__all__
        assert hasattr(generated, name)
        assert not hasattr(notebook_api, name)
    assert "build_records_preview" in progress.__all__
    assert hasattr(progress, "build_records_preview")


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
        module_lengths = {path.name: _line_count(path) for path in package.glob("*.py") if path.name != "__init__.py"}
        assert expected_modules.issubset(module_lengths)
        assert max(module_lengths.values()) <= 220


def test_campaign_progress_analysis_is_semantic_package_modules() -> None:
    progress_file = OPAL_SOURCE_ROOT / "analysis" / "campaign_progress.py"
    progress_package = OPAL_SOURCE_ROOT / "analysis" / "campaign_progress"

    assert not progress_file.exists()
    assert progress_package.is_dir()
    module_lengths = {
        path.name: _line_count(path) for path in progress_package.glob("*.py") if path.name != "__init__.py"
    }
    assert {"content.py", "ledger.py", "models.py", "records.py"}.issubset(module_lengths)
    assert max(module_lengths.values()) <= 180


def test_label_history_is_semantic_package_modules() -> None:
    history_file = OPAL_SOURCE_ROOT / "storage" / "label_history.py"
    history_package = OPAL_SOURCE_ROOT / "storage" / "label_history"

    assert not history_file.exists()
    assert history_package.is_dir()
    module_lengths = {
        path.name: _line_count(path) for path in history_package.glob("*.py") if path.name != "__init__.py"
    }
    assert module_lengths
    assert max(module_lengths.values()) <= 260


def test_ingest_y_command_is_semantic_package_modules() -> None:
    command_file = OPAL_SOURCE_ROOT / "cli" / "commands" / "ingest_y.py"
    command_package = OPAL_SOURCE_ROOT / "cli" / "commands" / "ingest_y"

    assert not command_file.exists()
    assert command_package.is_dir()
    module_lengths = {
        path.name: _line_count(path) for path in command_package.glob("*.py") if path.name != "__init__.py"
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
        path.name: _line_count(path) for path in stages_package.glob("*.py") if path.name != "__init__.py"
    }
    assert module_lengths
    assert max(module_lengths.values()) <= 260


def test_round_stages_package_imports_real_stage_entrypoints() -> None:
    module = importlib.import_module("dnadesign.opal.src.runtime.round.stages")

    assert module.stage_training.__name__ == "stage_training"
    assert module.stage_x_matrices.__name__ == "stage_x_matrices"
    assert module.stage_scoring.__name__ == "stage_scoring"
