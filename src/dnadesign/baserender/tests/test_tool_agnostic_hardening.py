"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/test_tool_agnostic_hardening.py

Static hardening tests that enforce tool-agnostic API and contract-driven parsing.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
from pathlib import Path

from dnadesign.baserender.src.render.renderer import renderer_descriptors


def _read(path: Path) -> str:
    return path.read_text().lower()


def test_public_api_module_avoids_tool_specific_input_assumptions() -> None:
    root = Path(__file__).resolve().parents[1]
    api_text = _read(root / "src" / "public" / "api.py")
    assert "densegen" not in api_text
    assert "sigma70" not in api_text


def test_job_parser_avoids_adapter_kind_branching() -> None:
    root = Path(__file__).resolve().parents[1]
    parser_text = _read(root / "src" / "config" / "render_job_v4.py")
    assert 'if kind == "densegen_tfbs"' not in parser_text
    assert 'if kind == "cruncher_best_window"' not in parser_text
    assert "densegen_tfbs" not in parser_text
    assert "cruncher_best_window" not in parser_text


def test_job_parser_avoids_transform_name_branching() -> None:
    root = Path(__file__).resolve().parents[1]
    parser_text = _read(root / "src" / "config" / "render_job_v4.py")
    for transform_name in (
        "attach_motifs_from_config",
        "attach_motifs_from_cruncher_lockfile",
        "attach_motifs_from_library",
        "sigma70",
    ):
        assert transform_name not in parser_text


def test_render_layer_does_not_import_producer_implementations() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "render"
    violations: list[str] = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports = (node.module,)
            else:
                continue
            for imported in imports:
                if imported.startswith("dnadesign.") and not imported.startswith(
                    ("dnadesign.baserender", "dnadesign.contracts")
                ):
                    violations.append(f"{path.relative_to(root)} imports {imported}")
    assert violations == []


def test_renderer_catalog_excludes_statistical_analysis_families() -> None:
    statistical_terms = {
        "dissimilarity",
        "distance",
        "heatmap",
        "metric",
        "objective",
        "rank",
        "scatter",
        "score",
        "similarity",
    }
    violations = [
        descriptor.name
        for descriptor in renderer_descriptors()
        if statistical_terms.intersection(descriptor.name.split("_"))
    ]
    assert violations == []
