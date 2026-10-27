"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/config/test_public_import_boundary.py

Import-boundary and public-surface contract tests for DenseGen.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import tomllib
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _densegen_runtime_root() -> Path:
    return Path(__file__).resolve().parents[2] / "src"


def test_densegen_runtime_does_not_import_cruncher_meme_parser_internal() -> None:
    disallowed = "dnadesign.cruncher.io.parsers.meme"
    violations: list[str] = []
    for path in sorted(_densegen_runtime_root().rglob("*.py")):
        if disallowed in path.read_text(encoding="utf-8"):
            violations.append(str(path))
    assert not violations, f"Found disallowed DenseGen->Cruncher parser internal imports: {violations}"


def test_notebook_template_uses_public_densegen_import_paths() -> None:
    notebook_template = _densegen_runtime_root() / "cli" / "notebook_template.py"
    text = notebook_template.read_text(encoding="utf-8")
    assert "dnadesign.densegen.src." not in text


def test_project_scripts_avoid_internal_src_module_paths() -> None:
    pyproject = _repo_root() / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    scripts = data["project"]["scripts"]
    offenders = {name: target for name, target in scripts.items() if ".src." in str(target)}
    assert not offenders, f"Project scripts must not expose internal '.src.' module paths: {offenders}"
