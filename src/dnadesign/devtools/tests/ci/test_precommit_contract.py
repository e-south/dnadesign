"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/ci/test_precommit_contract.py

Tests for repository-owned pre-commit toolchain contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    return next(parent for parent in current.parents if (parent / "pyproject.toml").exists())


def _precommit_config() -> dict:
    config_path = _repo_root() / ".pre-commit-config.yaml"
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def test_ruff_hooks_use_the_locked_project_executable() -> None:
    config = _precommit_config()
    repos = config["repos"]

    assert all(repo["repo"] != "https://github.com/astral-sh/ruff-pre-commit" for repo in repos)

    local_repo = next(repo for repo in repos if repo["repo"] == "local")
    hooks = {hook["id"]: hook for hook in local_repo["hooks"]}
    expected = {
        "ruff-check": "uv run --no-sync ruff check --fix",
        "ruff-format": "uv run --no-sync ruff format",
    }

    for hook_id, entry in expected.items():
        hook = hooks[hook_id]
        assert hook["entry"] == entry
        assert hook["language"] == "system"
        assert hook["types_or"] == ["python", "pyi", "jupyter"]
