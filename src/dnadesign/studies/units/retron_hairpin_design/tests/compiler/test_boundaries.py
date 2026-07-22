"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/compiler/test_boundaries.py

Compiler architecture and checked-in registry boundary tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
import json
import tomllib
from pathlib import Path

from dnadesign.studies.units.retron_hairpin_design.interfaces.cli.app import app

from ..support.cli import RUNNER
from ..support.compiler_fixtures import SCAR_NICK_HIT_LABELS
from ..support.paths import repo_root_from


def test_checked_in_registry_compiles_planned_scar_nick_hits(tmp_path: Path) -> None:
    repo_root = repo_root_from(__file__)
    study_dir = repo_root / "docs" / "studies" / "retron_hairpin_design"
    input_file = study_dir / "compiler" / "inputs" / "msd_design_hit_labels.txt"
    out_dir = tmp_path / "compiled"

    result = RUNNER.invoke(
        app,
        [
            "compile",
            "--input",
            input_file.as_posix(),
            "--study-dir",
            study_dir.as_posix(),
            "--allow-non-ligatable-s0",
            "--out-dir",
            out_dir.as_posix(),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["record_count"] == 18
    assert sorted(item.name for item in out_dir.iterdir()) == [
        "README.md",
        "manifest.json",
        "msd_design_catalog_v1.json",
        "reference_index.tsv",
        "references",
    ]
    reference_files = sorted((out_dir / "references").glob("*.msd_design_reference_v1.json"))
    assert len(reference_files) == 18
    assert all(path.is_file() for path in reference_files)
    assert not any(path.is_dir() for path in (out_dir / "references").iterdir())

    selected_labels = [
        line.strip()
        for line in input_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert selected_labels == SCAR_NICK_HIT_LABELS
    top_nick = [record for record in payload["records"] if record["scar_nick"]["nick_orientation"] == "top"]
    assert {record["construct_id"] for record in top_nick} == {"pES-retron-193", "pES-retron-194"}


def test_retron_msd_compiler_is_not_exposed_as_top_level_project_script() -> None:
    repo_root = repo_root_from(__file__)
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]

    assert "retron-msd" not in scripts
    assert all("retron_hairpin_design.interfaces.cli" not in target for target in scripts.values())


def test_retron_msd_study_uses_public_tool_apis_only() -> None:
    repo_root = repo_root_from(__file__)
    study_source = repo_root / "src" / "dnadesign" / "studies" / "units" / "retron_hairpin_design"
    study_paths = sorted(
        path for path in study_source.rglob("*.py") if "__pycache__" not in path.parts and "tests" not in path.parts
    )
    imports: set[str] = set()
    for path in study_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)

    assert "dnadesign.construct" in imports
    assert "dnadesign.baserender" in imports
    assert "dnadesign.construct.src.composition.runtime" not in imports
    assert not any(name.startswith("dnadesign.baserender.src") for name in imports)
    assert not any(name == "dnadesign.cruncher" or name.startswith("dnadesign.cruncher.src") for name in imports)
    assert not any(name.startswith("dnadesign.cruncher.workspaces") for name in imports)
    assert not any(name.startswith("dnadesign.folding.src") for name in imports)


def test_retron_msd_materialize_does_not_shell_out_to_inkscape() -> None:
    repo_root = repo_root_from(__file__)
    source_root = repo_root / "src" / "dnadesign" / "studies" / "units" / "retron_hairpin_design"
    compiler_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(source_root.rglob("*.py"))
        if "__pycache__" not in path.parts and "tests" not in path.parts
    )

    assert "inkscape" not in compiler_source.lower()
    assert "subprocess.run" not in compiler_source


def test_retron_msd_compiler_source_is_decomposed_by_responsibility() -> None:
    repo_root = repo_root_from(__file__)
    source_root = repo_root / "src" / "dnadesign" / "studies" / "units" / "retron_hairpin_design"
    budgets = {
        "compiler/references.py": 180,
        "compiler/catalog_bundle.py": 220,
        "compiler/materialization.py": 260,
        "compiler/exceptions.py": 60,
        "interfaces/cli/app.py": 360,
        "interfaces/cli/inputs.py": 140,
        "interfaces/cli/io.py": 140,
        "interfaces/cli/messages.py": 180,
        "interfaces/cli/review_outputs.py": 120,
        "catalog/compiler_spec.py": 450,
        "catalog/compiler_spec_io.py": 140,
        "catalog/specs/primitive_sources.py": 90,
        "catalog/specs/variant_metadata.py": 140,
        "catalog/strict_mapping_io.py": 120,
        "catalog/msd_ids.py": 450,
        "catalog/registry.py": 450,
        "artifact_contracts/composition_payload.py": 450,
        "artifact_contracts/output_guards.py": 450,
        "artifact_contracts/materialized_outputs.py": 450,
        "artifact_contracts/manifests.py": 450,
        "review_outputs/contracts/manifest.py": 140,
        "review_outputs/contracts/benchling_import.py": 140,
        "review_outputs/contracts/feature_directions.py": 80,
        "review_outputs/contracts/plan.py": 180,
        "review_outputs/contracts/record_ids.py": 60,
        "review_outputs/contracts/review_variant_ids.py": 130,
        "review_outputs/handoff/benchling.py": 190,
        "review_outputs/handoff/contract.py": 80,
        "review_outputs/handoff/genbank_features.py": 140,
        "review_outputs/handoff/index.py": 160,
        "review_outputs/pwm/baserender_record.py": 160,
        "review_outputs/pwm/logo.py": 200,
        "review_outputs/pwm/panel_labels.py": 60,
        "review_outputs/pwm/panel_metadata.py": 90,
        "review_outputs/pwm/retention.py": 240,
        "review_outputs/pwm/sequence_rows.py": 190,
        "review_outputs/pwm/triptych.py": 140,
        "review_outputs/pwm/trim_annotations.py": 80,
        "review_outputs/pwm/typography.py": 40,
        "review_outputs/pwm/visual_layers.py": 80,
        "review_outputs/sequence/evidence.py": 120,
        "review_outputs/sequence/index.py": 140,
        "review_outputs/sequence/variant_identity.py": 100,
        "review_outputs/service.py": 120,
        "review_outputs/video/frame_naming.py": 70,
        "review_outputs/video/montage.py": 170,
        "review_outputs/video/stills.py": 150,
    }

    for filename, max_lines in budgets.items():
        path = source_root / filename
        assert path.is_file(), filename
        line_count = _implementation_line_count(path)
        assert line_count <= max_lines, f"{filename} has {line_count} lines > {max_lines}"

    assert not (source_root / "catalog" / "primitive_sources.py").exists()
    assert not (source_root / "catalog" / "variant_metadata.py").exists()
    assert sorted(path.name for path in (source_root / "review_outputs").glob("*.py")) == ["__init__.py", "service.py"]
    assert not any((source_root / "review_outputs").glob("[ps]*_*.py"))
    assert not (source_root / "review_outputs" / "clone_handoff_index.py").exists()
    assert not (source_root / "review_outputs" / "handoff" / "clone_index.py").exists()


def test_retron_msd_compiler_tests_are_decomposed_by_responsibility() -> None:
    repo_root = repo_root_from(__file__)
    tests_root = repo_root / "src" / "dnadesign" / "studies" / "units" / "retron_hairpin_design" / "tests"
    budgets = {
        "compiler/test_cap_sources.py": 120,
        "compiler/test_msd_ids.py": 120,
        "compiler/test_cli_lint.py": 1000,
        "compiler/test_msd_unit.py": 120,
        "compiler/test_cli_compile.py": 280,
        "compiler/test_materialization.py": 900,
        "compiler/test_boundaries.py": 246,
        "compiler/specs/test_teto_trim_metadata.py": 140,
        "review_outputs/cli/fixtures.py": 70,
        "review_outputs/cli/test_review_outputs.py": 90,
        "review_outputs/cli/test_review_outputs_text.py": 70,
        "review_outputs/handoff/test_benchling_import.py": 110,
        "review_outputs/handoff/test_genbank_features.py": 60,
        "review_outputs/package/test_generation.py": 230,
        "review_outputs/package/test_review_variant_ids.py": 70,
        "review_outputs/package/test_validation_failures.py": 110,
        "review_outputs/pwm/test_retention.py": 100,
        "review_outputs/video/test_montage.py": 100,
        "review_outputs/video/test_review_still_quality.py": 110,
        "support/cli.py": 40,
        "support/compiler_fixtures.py": 80,
        "support/fake_genbank_features.py": 60,
        "support/pwm_fixtures.py": 70,
        "support/registry.py": 80,
        "support/review_ids.py": 40,
        "support/review_plans.py": 60,
        "support/review_outputs.py": 220,
        "support/viennarna.py": 100,
        "workbench/test_workbench_ia_contracts.py": 80,
    }

    for filename, max_lines in budgets.items():
        path = tests_root / filename
        assert path.is_file(), filename
        assert len(path.read_text(encoding="utf-8").splitlines()) <= max_lines

    assert not (tests_root / "compiler" / "test_msd_compiler.py").exists()
    assert sorted(path.name for path in (tests_root / "review_outputs").glob("test_*.py")) == []


def _implementation_line_count(path: Path) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) >= 10 and lines[0] == '"""' and lines[1] == "-" * 80 and lines[8] == "-" * 80 and lines[9] == '"""':
        return len(lines) - 10
    return len(lines)


def test_retron_msd_study_root_has_no_python_surface_modules() -> None:
    repo_root = repo_root_from(__file__)
    source_root = repo_root / "src" / "dnadesign" / "studies" / "units" / "retron_hairpin_design"

    top_level_py = sorted(path.name for path in source_root.glob("*.py"))

    assert top_level_py == ["__init__.py"]
    assert not (source_root / "cli.py").exists()
    assert not (source_root / "compiler.py").exists()
    assert not (source_root / "errors.py").exists()
