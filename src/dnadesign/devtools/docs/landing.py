"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/docs/landing.py

Reader-facing landing-page checks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def find_landing_readme_frontmatter_issues(repo_root: Path) -> list[str]:
    """Keep repository and tool landing pages free of machine metadata blocks."""

    paths = [repo_root / "README.md"]
    src_root = repo_root / "src" / "dnadesign"
    if src_root.exists():
        paths.extend(sorted(src_root.glob("*/README.md")))
    issues: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        if lines and lines[0] == "---":
            issues.append(f"{path}: landing README must start with reader-facing content, not YAML front matter.")
    return issues


__all__ = ["find_landing_readme_frontmatter_issues"]
