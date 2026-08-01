"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/docs/badges.py

Enforces the restrained Markdown badge surface used by repository docs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

MARKDOWN_INLINE_IMAGE_PATTERN = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<image>[^)\s]+)(?:\s+[^)]*)?\)")
MARKDOWN_REFERENCE_IMAGE_PATTERN = re.compile(r"!\[(?P<alt>[^\]]*)\]\[(?P<label>[^\]]*)\]")
MARKDOWN_SHORTCUT_REFERENCE_IMAGE_PATTERN = re.compile(r"!\[(?P<alt>[^\]]+)\](?![\[(])")
MARKDOWN_REFERENCE_DEFINITION_PATTERN = re.compile(
    r"^\[(?P<label>[^\]]+)\]:\s*<?(?P<target>[^\s>]+)>?",
    flags=re.MULTILINE,
)
HTML_IMAGE_PATTERN = re.compile(r"<img\b(?P<attributes>[^>]*)>", flags=re.IGNORECASE)
HTML_ATTRIBUTE_PATTERN = re.compile(
    r"(?P<name>[A-Za-z_:][-A-Za-z0-9_:.]*)\s*=\s*"
    r'(?:"(?P<double>[^"]*)"|\'(?P<single>[^\']*)\'|(?P<bare>[^\s"\'=<>`]+))'
)
BADGE_HINT_PATTERN = re.compile(
    r"(?:\bbadge\b|shields\.io|codecov\.io|\b(?:build|ci|coverage|codeql|license|release|security|status|version)\b)",
    flags=re.IGNORECASE,
)
ROOT_README_ALLOWED_BADGES = frozenset(
    {
        "[![CI](https://github.com/e-south/dnadesign/actions/workflows/ci.yaml/badge.svg?branch=main)]"
        "(https://github.com/e-south/dnadesign/actions/workflows/ci.yaml)",
        "[![Codecov](https://codecov.io/gh/e-south/dnadesign/graph/badge.svg)]"
        "(https://codecov.io/gh/e-south/dnadesign)",
        "[![MIT license](https://img.shields.io/badge/license-MIT-3D8068.svg)](LICENSE)",
    }
)


def _mask_fenced_code(content: str) -> str:
    masked: list[str] = []
    fence_character: str | None = None
    fence_length = 0
    for line in content.splitlines(keepends=True):
        candidate = line.lstrip(" ")
        indent = len(line) - len(candidate)
        body = candidate.rstrip("\r\n")
        fence = re.match(r"(?P<fence>`{3,}|~{3,})", body) if indent <= 3 else None
        inside_fence = fence_character is not None
        if inside_fence:
            closing = re.fullmatch(
                rf"{re.escape(fence_character)}{{{fence_length},}}[ \t]*",
                body,
            )
            masked.append("".join(character if character in "\r\n" else " " for character in line))
            if closing is not None:
                fence_character = None
                fence_length = 0
            continue
        if fence is not None:
            marker = fence.group("fence")
            fence_character = marker[0]
            fence_length = len(marker)
            masked.append("".join(character if character in "\r\n" else " " for character in line))
            continue
        masked.append(line)
    return "".join(masked)


def _html_attributes(raw_attributes: str) -> dict[str, str]:
    attributes: dict[str, str] = {}
    for match in HTML_ATTRIBUTE_PATTERN.finditer(raw_attributes):
        value = match.group("double") or match.group("single") or match.group("bare") or ""
        attributes[match.group("name").casefold()] = value
    return attributes


def _badge_image_positions(content: str) -> tuple[int, ...]:
    visible_content = _mask_fenced_code(content)
    reference_targets = {
        match.group("label").casefold(): match.group("target")
        for match in MARKDOWN_REFERENCE_DEFINITION_PATTERN.finditer(visible_content)
    }
    positions: set[int] = set()

    for match in MARKDOWN_INLINE_IMAGE_PATTERN.finditer(visible_content):
        if BADGE_HINT_PATTERN.search(f"{match.group('alt')} {match.group('image')}"):
            positions.add(match.start())

    for match in MARKDOWN_REFERENCE_IMAGE_PATTERN.finditer(visible_content):
        label = match.group("label") or match.group("alt")
        target = reference_targets.get(label.casefold(), "")
        if BADGE_HINT_PATTERN.search(f"{match.group('alt')} {label} {target}"):
            positions.add(match.start())

    for match in MARKDOWN_SHORTCUT_REFERENCE_IMAGE_PATTERN.finditer(visible_content):
        target = reference_targets.get(match.group("alt").casefold(), "")
        if target and BADGE_HINT_PATTERN.search(f"{match.group('alt')} {target}"):
            positions.add(match.start())

    for match in HTML_IMAGE_PATTERN.finditer(visible_content):
        attributes = _html_attributes(match.group("attributes"))
        if BADGE_HINT_PATTERN.search(f"{attributes.get('alt', '')} {attributes.get('src', '')}"):
            positions.add(match.start())

    return tuple(sorted(positions))


def find_markdown_badge_policy_issues(repo_root: Path, markdown_files: Iterable[Path]) -> list[str]:
    """Return badge-policy violations without changing documentation."""
    root_readme = (repo_root / "README.md").resolve()
    issues: list[str] = []
    reported_locations: set[tuple[Path, int]] = set()
    root_badge_counts: dict[str, int] = {}
    for path in markdown_files:
        content = path.read_text(encoding="utf-8")
        lines = content.splitlines()
        for position in _badge_image_positions(content):
            line_no = content[:position].count("\n") + 1
            location = (path, line_no)
            if location in reported_locations:
                continue
            reported_locations.add(location)
            line = lines[line_no - 1].strip()
            if path.resolve() != root_readme:
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
