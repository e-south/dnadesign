"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/package/test_layout.py

Package layout contract tests for devtools domain decomposition.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
from pathlib import Path

_HEADER_SEPARATOR = "-" * 80
_FORBIDDEN_HEADER_AUTHOR_TOKENS = ("Codex", "ChatGPT", "OpenAI", "Open AI")
_HEADER_EXCLUDED_PARTS = {
    ".venv",
    "__pycache__",
    "archived",
    "batch_results",
    "prototypes",
    "runs",
}
_ELM_HEADER_EXCEPTIONS = {
    "src/dnadesign/opal/src/models/gaussian_process.py",
    "src/dnadesign/opal/src/objectives/sfxi_math.py",
    "src/dnadesign/opal/src/objectives/sfxi_v1.py",
    "src/dnadesign/opal/src/selection/expected_improvement.py",
    "src/dnadesign/opal/src/selection/top_n.py",
    "src/dnadesign/opal/src/transforms_y/intensity_median_iqr.py",
}


def _devtools_root() -> Path:
    current = Path(__file__).resolve()
    return next(
        parent
        for parent in current.parents
        if (parent / "tests" / "support").is_dir() and (parent / "__init__.py").exists()
    )


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    return next(parent for parent in current.parents if (parent / "pyproject.toml").exists())


def _is_generated_marimo_notebook(lines: list[str]) -> bool:
    return (
        bool(lines) and lines[0] == "import marimo" and any(line.startswith("__generated_with") for line in lines[:5])
    )


def test_devtools_domain_packages_importable() -> None:
    assert importlib.import_module("dnadesign.devtools.architecture.boundaries")
    assert importlib.import_module("dnadesign.devtools.ci.changed_files")
    assert importlib.import_module("dnadesign.devtools.ci.changes")
    assert importlib.import_module("dnadesign.devtools.ci.test_targets")
    assert importlib.import_module("dnadesign.devtools.docs.checks")
    assert importlib.import_module("dnadesign.devtools.docs.runbook_catalog")
    assert importlib.import_module("dnadesign.devtools.quality.coverage_summary")
    assert importlib.import_module("dnadesign.devtools.quality.entropy")
    assert importlib.import_module("dnadesign.devtools.quality.score")
    assert importlib.import_module("dnadesign.devtools.quality.tool_coverage")
    assert importlib.import_module("dnadesign.devtools.runtime.meme_env")
    assert importlib.import_module("dnadesign.devtools.runtime.pytest_gate")
    assert importlib.import_module("dnadesign.devtools.security.secrets_baseline")
    assert importlib.import_module("dnadesign.devtools.security.tracked_text_privacy")


def test_devtools_canonical_entrypoints_have_main() -> None:
    canonical_entrypoints = [
        "dnadesign.devtools.architecture.boundaries",
        "dnadesign.devtools.ci.changed_files",
        "dnadesign.devtools.ci.changes",
        "dnadesign.devtools.ci.test_targets",
        "dnadesign.devtools.quality.coverage_summary",
        "dnadesign.devtools.docs.checks",
        "dnadesign.devtools.docs.runbook_catalog",
        "dnadesign.devtools.runtime.meme_env",
        "dnadesign.devtools.runtime.pytest_gate",
        "dnadesign.devtools.quality.entropy",
        "dnadesign.devtools.quality.score",
        "dnadesign.devtools.security.secrets_baseline",
        "dnadesign.devtools.security.tracked_text_privacy",
        "dnadesign.devtools.quality.tool_coverage",
    ]
    for module_path in canonical_entrypoints:
        assert callable(importlib.import_module(module_path).main)


def test_devtools_top_level_contains_no_tool_modules() -> None:
    devtools_root = _devtools_root()
    top_level_modules = {path.name for path in devtools_root.glob("*.py")}
    assert top_level_modules == {"__init__.py"}
    assert not (devtools_root / "testsupport").exists()


def test_devtools_tests_are_grouped_by_runtime_domain() -> None:
    tests_root = _devtools_root() / "tests"
    expected_domains = {
        "architecture",
        "ci",
        "docs",
        "package",
        "quality",
        "runtime",
        "security",
        "support",
    }
    assert {path.name for path in tests_root.iterdir() if path.is_dir()} >= expected_domains


def test_active_python_module_headers_are_canonical() -> None:
    repo_root = _repo_root()
    dnadesign_root = repo_root / "src" / "dnadesign"
    invalid: dict[str, str] = {}
    elm_seen: set[str] = set()

    for path in sorted(dnadesign_root.rglob("*.py")):
        relative_parts = set(path.relative_to(dnadesign_root).parts)
        if relative_parts & _HEADER_EXCLUDED_PARTS:
            continue
        relative_path = path.relative_to(repo_root).as_posix()
        lines = path.read_text(encoding="utf-8").splitlines()
        if _is_generated_marimo_notebook(lines):
            continue
        header = "\n".join(lines[:12])
        if any(token in header for token in _FORBIDDEN_HEADER_AUTHOR_TOKENS):
            invalid[relative_path] = "agent author claim is not allowed"
            continue
        if "Elm Markert" in header:
            elm_seen.add(relative_path)
            if relative_path not in _ELM_HEADER_EXCEPTIONS:
                invalid[relative_path] = "unexpected Elm-authored header exception"
            continue
        if len(lines) < 10:
            invalid[relative_path] = "missing canonical header"
            continue
        if lines[0] != '"""' or lines[1] != _HEADER_SEPARATOR or lines[2] != "dnadesign":
            invalid[relative_path] = "header prefix mismatch"
            continue
        if lines[3] != relative_path:
            invalid[relative_path] = f"path mismatch: {lines[3]!r}"
            continue
        if lines[4] != "" or not lines[5].strip() or len(lines[5]) > 140:
            invalid[relative_path] = "missing concise module purpose"
            continue
        if lines[6] != "" or lines[7] != "Module Author(s): Eric J. South":
            invalid[relative_path] = "author mismatch"
            continue
        if lines[8] != _HEADER_SEPARATOR or lines[9] != '"""':
            invalid[relative_path] = "header suffix mismatch"
            continue

    assert invalid == {}
    assert elm_seen == _ELM_HEADER_EXCEPTIONS
