"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/densegen_axis_probe/test_package_contracts.py

Regression tests for package studies units stress ethanol cipro growth decision.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .helpers import (
    Path,
    importlib,
    subprocess,
    sys,
)
from .probe_modules import PROBE_PACKAGE


def _line_count(path: Path) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) >= 10 and lines[0] == '"""' and lines[1] == "-" * 80 and lines[8] == "-" * 80 and lines[9] == '"""':
        return len(lines) - 10
    return len(lines)


def test_probe_package_root_exports_no_flat_api_surface() -> None:
    package = importlib.import_module(PROBE_PACKAGE)

    assert package.__all__ == []
    assert "build_axis_oracle" not in vars(package)
    assert "main" not in vars(package)


def test_cli_import_keeps_status_path_run_stack_lazy() -> None:
    module = f"{PROBE_PACKAGE}.cli"
    script = (
        "import sys; "
        f"import {module}; "
        "heavy = sorted(name for name in ('numpy', 'pandas', 'pyarrow', 'yaml') if name in sys.modules); "
        "print(heavy); "
        "raise SystemExit(1 if heavy else 0)"
    )

    result = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True)

    assert result.returncode == 0, result.stdout + result.stderr


def test_probe_review_is_semantic_package() -> None:
    probe_root = Path("src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe")
    review_file = probe_root / "review.py"
    review_package = probe_root / "reporting" / "review"

    assert not review_file.exists()
    assert not (review_package / "probe_plots.py").exists()
    assert review_package.is_dir()
    assert (review_package / "aggregate_plots").is_dir()
    module_lengths = {
        path.relative_to(review_package).as_posix(): _line_count(path)
        for path in review_package.rglob("*.py")
        if path.name != "__init__.py"
    }
    assert module_lengths
    assert max(module_lengths.values()) <= 360


def test_probe_root_keeps_only_entrypoint_modules() -> None:
    probe_root = Path("src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe")

    assert {path.name for path in probe_root.glob("*.py")} == {"__init__.py", "__main__.py", "cli.py"}


def test_probe_known_large_entrypoints_are_explicitly_budget_guarded() -> None:
    probe_root = Path("src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe")
    budgets = {
        probe_root / "tfbs" / "cli.py": 480,
        probe_root / "plan_logic" / "axis_oracle.py": 470,
    }

    for path, max_lines in budgets.items():
        assert path.is_file()
        assert _line_count(path) <= max_lines


def test_tfbs_null_and_slot_diagnostics_are_semantic_packages() -> None:
    probe_root = Path("src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe")
    tfbs_root = probe_root / "tfbs"
    nulls_package = tfbs_root / "nulls"
    candidate_scopes_package = tfbs_root / "candidate_scopes"
    slot_diagnostics_package = tfbs_root / "stage_b" / "slot_diagnostics"

    assert not (tfbs_root / "nulls.py").exists()
    assert nulls_package.is_dir()
    assert not (tfbs_root / "candidate_scopes.py").exists()
    assert candidate_scopes_package.is_dir()
    assert not (tfbs_root / "stage_b" / "slot_diagnostics.py").exists()
    assert not (tfbs_root / "stage_b" / "slot_plots.py").exists()
    assert slot_diagnostics_package.is_dir()
    assert (slot_diagnostics_package / "plots").is_dir()

    module_lengths = {
        path.relative_to(probe_root).as_posix(): _line_count(path)
        for package in (
            nulls_package,
            candidate_scopes_package,
            slot_diagnostics_package,
            slot_diagnostics_package / "plots",
        )
        for path in package.glob("*.py")
        if path.name != "__init__.py"
    }
    assert module_lengths
    assert max(module_lengths.values()) <= 360


def test_tfbs_stage_b_configs_and_review_are_semantic_packages() -> None:
    probe_root = Path("src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe")
    stage_b_root = probe_root / "tfbs" / "stage_b"
    semantic_packages = (
        stage_b_root / "configs",
        stage_b_root / "execution",
        stage_b_root / "review",
        stage_b_root / "review" / "plots",
        stage_b_root / "notebook_visuals",
        stage_b_root / "learning_loop_baselines",
        stage_b_root / "learning_loop_baselines" / "plots",
    )

    assert not (stage_b_root / "configs.py").exists()
    assert not (stage_b_root / "execution.py").exists()
    assert not (stage_b_root / "review.py").exists()
    assert not (stage_b_root / "review_plots.py").exists()
    assert not (stage_b_root / "notebook_visuals.py").exists()
    assert not (stage_b_root / "notebook_visual_specs.py").exists()
    assert not (stage_b_root / "learning_loop_baselines.py").exists()
    assert all(path.is_dir() for path in semantic_packages)

    module_lengths = {
        path.relative_to(probe_root).as_posix(): _line_count(path)
        for package in semantic_packages
        for path in package.glob("*.py")
        if path.name != "__init__.py"
    }
    assert module_lengths
    assert max(module_lengths.values()) <= 360


def test_tfbs_target_profiles_are_first_class_contract_module() -> None:
    probe_root = Path("src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/densegen_axis_probe")
    tfbs_root = probe_root / "tfbs"

    assert (tfbs_root / "profiles.py").exists()
    assert not (tfbs_root / "target_profiles.py").exists()


def test_probe_tests_are_semantic_package() -> None:
    tests_root = Path("src/dnadesign/studies/tests")
    legacy_flat_file = tests_root / "test_stress_ethanol_cipro_opal_densegen_axis_probe.py"
    test_package = (
        Path("src/dnadesign/studies/units")
        / "stress_ethanol_cipro_growth"
        / "tests"
        / "decision"
        / "opal"
        / "densegen_axis_probe"
    )

    assert not legacy_flat_file.exists()
    assert not (tests_root / "stress_ethanol_cipro_growth").exists()
    assert test_package.is_dir()
    module_lengths = {path.name: _line_count(path) for path in test_package.glob("*.py") if path.name != "__init__.py"}
    assert module_lengths
    assert max(module_lengths.values()) <= 360
