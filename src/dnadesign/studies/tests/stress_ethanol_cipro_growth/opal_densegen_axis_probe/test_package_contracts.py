from __future__ import annotations

from .helpers import (
    Path,
    importlib,
    subprocess,
    sys,
)


def test_probe_package_root_exports_no_flat_api_surface() -> None:
    package = importlib.import_module("dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe")

    assert package.__all__ == []
    assert "build_axis_oracle" not in vars(package)
    assert "main" not in vars(package)


def test_cli_import_keeps_status_path_run_stack_lazy() -> None:
    module = "dnadesign.studies.studies.stress_ethanol_cipro_growth.opal_densegen_axis_probe.cli"
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
    probe_root = Path("src/dnadesign/studies/studies/stress_ethanol_cipro_growth/opal_densegen_axis_probe")
    review_file = probe_root / "review.py"
    review_package = probe_root / "review"

    assert not review_file.exists()
    assert review_package.is_dir()
    module_lengths = {
        path.relative_to(review_package).as_posix(): len(path.read_text(encoding="utf-8").splitlines())
        for path in review_package.rglob("*.py")
        if path.name != "__init__.py"
    }
    assert module_lengths
    assert max(module_lengths.values()) <= 360


def test_probe_tests_are_semantic_package() -> None:
    tests_root = Path("src/dnadesign/studies/tests")
    legacy_flat_file = tests_root / "test_stress_ethanol_cipro_opal_densegen_axis_probe.py"
    test_package = tests_root / "stress_ethanol_cipro_growth" / "opal_densegen_axis_probe"

    assert not legacy_flat_file.exists()
    assert test_package.is_dir()
    module_lengths = {
        path.name: len(path.read_text(encoding="utf-8").splitlines())
        for path in test_package.glob("*.py")
        if path.name != "__init__.py"
    }
    assert module_lengths
    assert max(module_lengths.values()) <= 360
