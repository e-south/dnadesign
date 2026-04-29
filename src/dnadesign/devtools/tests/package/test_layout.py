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


def _devtools_root() -> Path:
    current = Path(__file__).resolve()
    return next(
        parent
        for parent in current.parents
        if (parent / "tests" / "support").is_dir() and (parent / "__init__.py").exists()
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
