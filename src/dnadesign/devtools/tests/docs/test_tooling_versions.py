"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/docs/test_tooling_versions.py

Keeps documented tool-version requirements aligned with repository policy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import tomllib
from pathlib import Path


def test_uv_requirement_is_identical_in_policy_and_setup_docs() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    policy = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    required = policy["tool"]["uv"]["required-version"]

    assert required == ">=0.12.3,<0.13"
    for path in (repo_root / "docs/setup/installation.md", repo_root / "docs/setup/dependencies.md"):
        assert f"`{required}`" in path.read_text(encoding="utf-8")
