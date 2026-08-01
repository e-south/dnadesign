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
HTML_COMMENT_PATTERN = re.compile(r"<!--[\s\S]*?-->")
HTML_ANCHOR_OPEN_PATTERN = re.compile(r"<a(?:\s|>)", flags=re.IGNORECASE)
BADGE_SOURCE_PATTERN = re.compile(
    r"(?:shields\.io|codecov\.io|(?:^|[/_.-])badge(?:[./?_-]|$))",
    flags=re.IGNORECASE,
)
BADGE_LABEL_PATTERN = re.compile(r"\s*(?:ci|coverage|codecov|license)\s*", flags=re.IGNORECASE)
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


def _mask_range(content: str, start: int, end: int) -> str:
    masked_range = "".join(character if character in "\r\n" else " " for character in content[start:end])
    return content[:start] + masked_range + content[end:]


def _mask_html_comments(content: str) -> str:
    masked = content
    for match in reversed(tuple(HTML_COMMENT_PATTERN.finditer(content))):
        masked = _mask_range(masked, match.start(), match.end())
    return masked


def _mask_inline_code(content: str) -> str:
    masked = content
    cursor = 0
    while cursor < len(content):
        if content[cursor] != "`":
            cursor += 1
            continue
        opening_end = cursor
        while opening_end < len(content) and content[opening_end] == "`":
            opening_end += 1
        delimiter = content[cursor:opening_end]
        search_from = opening_end
        closing_start = -1
        while True:
            candidate = content.find(delimiter, search_from)
            if candidate < 0:
                break
            candidate_end = candidate + len(delimiter)
            if (candidate == 0 or content[candidate - 1] != "`") and (
                candidate_end == len(content) or content[candidate_end] != "`"
            ):
                closing_start = candidate
                break
            search_from = candidate_end
        if closing_start < 0:
            cursor = opening_end
            continue
        closing_end = closing_start + len(delimiter)
        masked = _mask_range(masked, cursor, closing_end)
        cursor = closing_end
    return masked


def _mask_non_rendered_spans(content: str) -> str:
    return _mask_inline_code(_mask_html_comments(_mask_fenced_code(content)))


def _html_attributes(raw_attributes: str) -> dict[str, str]:
    attributes: dict[str, str] = {}
    for match in HTML_ATTRIBUTE_PATTERN.finditer(raw_attributes):
        value = match.group("double") or match.group("single") or match.group("bare") or ""
        attributes[match.group("name").casefold()] = value
    return attributes


def _is_markdown_image_linked(content: str, position: int) -> bool:
    return position > 0 and content[position - 1] == "["


def _looks_like_badge(*, label: str, source: str, linked: bool) -> bool:
    return BADGE_SOURCE_PATTERN.search(source) is not None or (
        linked and BADGE_LABEL_PATTERN.fullmatch(label) is not None
    )


def _is_html_image_linked(content: str, start: int, end: int) -> bool:
    lowered = content.casefold()
    openings = tuple(HTML_ANCHOR_OPEN_PATTERN.finditer(content, 0, start))
    opening = openings[-1].start() if openings else -1
    prior_closing = lowered.rfind("</a>", 0, start)
    next_closing = lowered.find("</a>", end)
    return opening > prior_closing and next_closing >= 0


def _badge_image_positions(content: str) -> tuple[int, ...]:
    visible_content = _mask_non_rendered_spans(content)
    reference_targets = {
        match.group("label").casefold(): match.group("target")
        for match in MARKDOWN_REFERENCE_DEFINITION_PATTERN.finditer(visible_content)
    }
    positions: set[int] = set()

    for match in MARKDOWN_INLINE_IMAGE_PATTERN.finditer(visible_content):
        if _looks_like_badge(
            label=match.group("alt"),
            source=match.group("image"),
            linked=_is_markdown_image_linked(visible_content, match.start()),
        ):
            positions.add(match.start())

    for match in MARKDOWN_REFERENCE_IMAGE_PATTERN.finditer(visible_content):
        label = match.group("label") or match.group("alt")
        target = reference_targets.get(label.casefold(), "")
        if _looks_like_badge(
            label=match.group("alt"),
            source=target,
            linked=_is_markdown_image_linked(visible_content, match.start()),
        ):
            positions.add(match.start())

    for match in MARKDOWN_SHORTCUT_REFERENCE_IMAGE_PATTERN.finditer(visible_content):
        target = reference_targets.get(match.group("alt").casefold(), "")
        if target and _looks_like_badge(
            label=match.group("alt"),
            source=target,
            linked=_is_markdown_image_linked(visible_content, match.start()),
        ):
            positions.add(match.start())

    for match in HTML_IMAGE_PATTERN.finditer(visible_content):
        attributes = _html_attributes(match.group("attributes"))
        if _looks_like_badge(
            label=attributes.get("alt", ""),
            source=attributes.get("src", ""),
            linked=_is_html_image_linked(visible_content, match.start(), match.end()),
        ):
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
