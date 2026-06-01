from __future__ import annotations

from pathlib import Path


def _study_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if parent.name == "stress_ethanol_cipro_growth":
            return parent
    raise RuntimeError("stress_ethanol_cipro_growth root not found")


def test_study_root_uses_typed_ontology_directories() -> None:
    study_root = _study_root()
    top_level_dirs = {path.name for path in study_root.iterdir() if path.is_dir()}

    assert {"decision", "operations", "tests", "workbench"}.issubset(top_level_dirs)
    assert not {
        "deliverables",
        "notes",
        "opal_batch0",
        "opal_densegen_axis_probe",
        "reference_sets",
        "status",
    }.intersection(top_level_dirs)
    assert not (study_root / "study.yaml").exists()


def test_opal_and_status_surfaces_have_explicit_owner_paths() -> None:
    study_root = _study_root()

    assert (study_root / "decision" / "opal" / "batch0" / "sampling.yaml").is_file()
    assert (study_root / "decision" / "opal" / "densegen_axis_probe" / "cli.py").is_file()
    assert (study_root / "operations" / "status" / "ops" / "status.registry.yaml").is_file()
    assert (study_root / "workbench" / "study.yaml").is_file()
    assert (study_root / "workbench" / "deliverables" / "README.md").is_file()
    assert (study_root / "workbench" / "notes" / "README.md").is_file()
    assert (study_root / "workbench" / "reference_sets" / "promoter_wt_core.yaml").is_file()


def test_tests_mirror_source_ontology() -> None:
    tests_root = _study_root() / "tests"

    assert (tests_root / "decision" / "opal" / "batch0").is_dir()
    assert (tests_root / "decision" / "opal" / "densegen_axis_probe").is_dir()
    assert (tests_root / "operations" / "status").is_dir()
    assert not (tests_root / "opal_batch0").exists()
    assert not (tests_root / "opal_densegen_axis_probe").exists()
    assert not (tests_root / "status").exists()
