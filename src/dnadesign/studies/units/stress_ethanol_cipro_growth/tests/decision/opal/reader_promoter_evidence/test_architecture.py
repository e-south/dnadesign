"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/reader_promoter_evidence/test_architecture.py

Enforce architecture boundaries for the study-owned Reader evidence adapter.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path("src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/reader_promoter_evidence")


@pytest.mark.parametrize("module_path", sorted(PACKAGE_ROOT.glob("*.py")), ids=lambda path: path.name)
def test_reader_evidence_modules_stay_bounded(module_path: Path) -> None:
    assert len(module_path.read_text(encoding="utf-8").splitlines()) <= 320


def test_study_adapter_does_not_import_reader_or_generic_opal_internals() -> None:
    reader_roots = {"reader", "reader_workbench"}
    reader_imports: list[str] = []
    source_parts: list[str] = []
    for path in PACKAGE_ROOT.glob("*.py"):
        source_part = path.read_text(encoding="utf-8")
        source_parts.append(source_part)
        for node in ast.walk(ast.parse(source_part, filename=str(path))):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            reader_imports.extend(
                f"{path.name}:{node.lineno}:{module}"
                for module in modules
                if module.split(".", maxsplit=1)[0] in reader_roots
            )
    source = "\n".join(source_parts)

    assert reader_imports == []
    assert "dnadesign.opal.src" not in source


def test_study_owns_notebook_descriptor_and_semantic_verifier() -> None:
    adapter = (PACKAGE_ROOT / "notebook_adapter.py").read_text(encoding="utf-8")
    generic_api = Path("src/dnadesign/opal/api/reader_evidence.py").read_text(encoding="utf-8")
    generic_visual = Path("src/dnadesign/opal/src/analysis/notebook_components/reader_evidence_visual.py").read_text(
        encoding="utf-8"
    )
    generic_media = Path("src/dnadesign/opal/src/analysis/notebook_components/reader_evidence_media.py").read_text(
        encoding="utf-8"
    )
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "register_reader_evidence_artifact_adapter" in adapter
    assert "verify_reader_promoter_evidence_manifest" in adapter
    assert "promoter_response_evidence" not in generic_api
    assert "response_window" not in generic_api
    assert "promoter" not in generic_visual.lower()
    assert "promoter_response_evidence" not in generic_media
    assert "sfxi_vec8_heatmap" not in generic_media
    assert '[project.entry-points."dnadesign.opal.reader_evidence_artifacts"]' in pyproject
    assert "reader_promoter_evidence.notebook_adapter:register_notebook_adapter" in pyproject
    assert not Path("src/dnadesign/opal/src/analysis/notebook_components/reader_promoter_evidence_contract.py").exists()
