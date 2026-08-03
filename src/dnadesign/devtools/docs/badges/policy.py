"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/docs/badges/policy.py

Apply the repository's restrained Markdown badge policy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from dnadesign.devtools.docs.badges.detection import looks_like_badge
from dnadesign.devtools.docs.badges.images import ImageOccurrence, rendered_image_occurrences

ROOT_README_ALLOWED_BADGES = frozenset(
    {
        "[![CI](https://github.com/e-south/dnadesign/actions/workflows/ci.yaml/badge.svg?branch=main)]"
        "(https://github.com/e-south/dnadesign/actions/workflows/ci.yaml)",
        "[![Codecov](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg)]"
        "(https://codecov.io/gh/e-south/dnadesign)",
        "[![MIT license](https://img.shields.io/badge/license-MIT-3D8068.svg)](LICENSE)",
    }
)


def _rendered_badge_occurrences(content: str) -> tuple[ImageOccurrence, ...]:
    return tuple(
        occurrence
        for occurrence in rendered_image_occurrences(content)
        if looks_like_badge(
            label=occurrence.spec.label,
            sources=occurrence.spec.sources,
            linked=occurrence.spec.linked,
        )
    )


def rendered_markdown_badge_lines(content: str) -> tuple[str, ...]:
    """Return rendered badge lines after excluding literal Markdown examples."""

    lines = content.splitlines()
    return tuple(lines[occurrence.line_no - 1].strip() for occurrence in _rendered_badge_occurrences(content))


def find_markdown_badge_policy_issues(repo_root: Path, markdown_files: Iterable[Path]) -> list[str]:
    """Return badge-policy violations without changing documentation."""

    root_readme = (repo_root / "README.md").resolve()
    issues: list[str] = []
    root_badge_counts: dict[str, int] = {}
    for path in markdown_files:
        content = path.read_text(encoding="utf-8")
        lines = content.splitlines()
        is_root_readme = path.resolve() == root_readme
        for occurrence in _rendered_badge_occurrences(content):
            line_no = occurrence.line_no
            line = lines[line_no - 1].strip()
            if not is_root_readme:
                issues.append(
                    f"{path}:{line_no}: badges belong only in the root README; use a plain text link instead."
                )
                continue
            if line not in ROOT_README_ALLOWED_BADGES:
                issues.append(
                    f"{path}:{line_no}: root README badge is not in the restrained CI, coverage, and license set."
                )
                continue
            root_badge_counts[line] = root_badge_counts.get(line, 0) + 1
            if root_badge_counts[line] > 1:
                issues.append(f"{path}:{line_no}: root README must include each approved badge at most once.")
    return issues
