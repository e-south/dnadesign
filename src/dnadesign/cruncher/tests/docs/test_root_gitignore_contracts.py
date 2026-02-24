"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/docs/test_root_gitignore_contracts.py

Validate root .gitignore contracts needed for clean Cruncher workspace/docs UX.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def test_gitignore_does_not_hide_cruncher_docs_by_runtime_globs() -> None:
    text = (_repo_root() / ".gitignore").read_text()
    assert "src/dnadesign/cruncher/**/elites.*" not in text
    assert "src/dnadesign/cruncher/**/report.md" not in text
    assert "src/dnadesign/cruncher/**/report.json" not in text


def test_gitignore_allows_cruncher_docs_images() -> None:
    text = (_repo_root() / ".gitignore").read_text()
    assert "*.png" in text
    assert "!src/dnadesign/cruncher/docs/**/*.png" in text
    assert "!src/dnadesign/cruncher/docs/**/*.jpg" in text
    assert "!src/dnadesign/cruncher/docs/**/*.jpeg" in text
    assert "!src/dnadesign/cruncher/docs/**/*.gif" in text


def test_gitignore_uses_repo_shared_matplotlib_cache_contract() -> None:
    text = (_repo_root() / ".gitignore").read_text()
    assert ".cache/matplotlib/" in text
    assert "**/.mplcache/" not in text
